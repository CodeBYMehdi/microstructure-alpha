"""Webhook alerting — Slack, Telegram, Discord integration.

Sends critical alerts to external messaging platforms so the trader
gets notified when things go wrong (kill switch, connectivity loss,
drawdown breach, etc.)

Usage:
    alerter = WebhookAlerter()
    alerter.add_slack("https://hooks.slack.com/services/XXX/YYY/ZZZ")
    alerter.add_telegram(bot_token="xxx", chat_id="12345")
    
    alerter.send_alert(
        title="🚨 Kill Switch Triggered",
        message="Drawdown exceeded 5% limit. All orders cancelled.",
        severity="CRITICAL",
    )

Can also be configured via environment variables:
    ALERT_SLACK_WEBHOOK=https://hooks.slack.com/...
    ALERT_TELEGRAM_TOKEN=xxx
    ALERT_TELEGRAM_CHAT_ID=xxx
    ALERT_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
"""

import json
import logging
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque
from urllib import request as urllib_request
from urllib.error import URLError
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    timestamp: str
    title: str
    message: str
    severity: str
    channels_sent: List[str]
    success: bool
    error: Optional[str] = None


class WebhookAlerter:
    """Multi-channel webhook alerting system."""

    def __init__(
        self,
        cooldown_seconds: float = 60.0,    # Min time between same-type alerts
        max_alerts_per_hour: int = 30,      # Rate limit
        auto_configure: bool = True,        # Read from env vars
    ):
        self.cooldown = cooldown_seconds
        self.max_per_hour = max_alerts_per_hour

        # Channel configs
        self._slack_webhooks: List[str] = []
        self._telegram_configs: List[Dict] = []  # [{"token": ..., "chat_id": ...}]
        self._discord_webhooks: List[str] = []

        # Rate limiting
        self._last_alert_by_type: Dict[str, float] = {}
        self._alerts_this_hour: Deque[float] = deque(maxlen=max_alerts_per_hour)
        self._history: Deque[AlertRecord] = deque(maxlen=500)

        # Background send queue
        self._send_queue: List[Dict] = []
        self._lock = threading.Lock()

        if auto_configure:
            self._configure_from_env()

        n_channels = len(self._slack_webhooks) + len(self._telegram_configs) + len(self._discord_webhooks)
        logger.info(f"WebhookAlerter initialized: {n_channels} channel(s) configured")

    def _configure_from_env(self) -> None:
        """Auto-configure from environment variables."""
        slack = os.environ.get('ALERT_SLACK_WEBHOOK')
        if slack:
            self._slack_webhooks.append(slack)

        tg_token = os.environ.get('ALERT_TELEGRAM_TOKEN')
        tg_chat = os.environ.get('ALERT_TELEGRAM_CHAT_ID')
        if tg_token and tg_chat:
            self._telegram_configs.append({'token': tg_token, 'chat_id': tg_chat})

        discord = os.environ.get('ALERT_DISCORD_WEBHOOK')
        if discord:
            self._discord_webhooks.append(discord)

    # ── Channel Registration ──

    def add_slack(self, webhook_url: str) -> None:
        self._slack_webhooks.append(webhook_url)
        logger.info("Slack webhook added")

    def add_telegram(self, bot_token: str, chat_id: str) -> None:
        self._telegram_configs.append({'token': bot_token, 'chat_id': chat_id})
        logger.info("Telegram webhook added")

    def add_discord(self, webhook_url: str) -> None:
        self._discord_webhooks.append(webhook_url)
        logger.info("Discord webhook added")

    # ── Alert Sending ──

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "WARNING",   # INFO, WARNING, CRITICAL
        alert_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Send an alert to all configured channels.
        
        Returns True if at least one channel succeeded.
        """
        alert_type = alert_type or title

        # Rate limiting
        if not self._check_rate_limit(alert_type):
            logger.debug(f"Alert rate-limited: {alert_type}")
            return False

        # Build formatted message
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(severity, "📢")

        full_message = f"{emoji} **{title}**\n{message}"
        if metadata:
            meta_str = "\n".join(f"  • {k}: {v}" for k, v in metadata.items())
            full_message += f"\n\n{meta_str}"
        full_message += f"\n\n_({timestamp})_"

        channels_sent = []
        errors = []

        # Send to all channels (in background thread to not block trading)
        def _send():
            nonlocal channels_sent, errors

            # Slack
            for url in self._slack_webhooks:
                try:
                    self._send_slack(url, title, full_message, severity)
                    channels_sent.append("slack")
                except Exception as e:
                    errors.append(f"slack: {e}")

            # Telegram
            for config in self._telegram_configs:
                try:
                    self._send_telegram(config['token'], config['chat_id'], full_message)
                    channels_sent.append("telegram")
                except Exception as e:
                    errors.append(f"telegram: {e}")

            # Discord
            for url in self._discord_webhooks:
                try:
                    self._send_discord(url, title, full_message, severity)
                    channels_sent.append("discord")
                except Exception as e:
                    errors.append(f"discord: {e}")

            # Record
            record = AlertRecord(
                timestamp=timestamp,
                title=title,
                message=message,
                severity=severity,
                channels_sent=channels_sent,
                success=len(channels_sent) > 0,
                error="; ".join(errors) if errors else None,
            )
            with self._lock:
                self._history.append(record)

            if errors:
                logger.warning(f"Alert send errors: {errors}")

        thread = threading.Thread(target=_send, daemon=True, name="alert-send")
        thread.start()

        # Also log locally
        log_fn = logger.critical if severity == "CRITICAL" else (logger.warning if severity == "WARNING" else logger.info)
        log_fn(f"ALERT [{severity}] {title}: {message}")

        # Track for cooldown
        self._last_alert_by_type[alert_type] = time.time()
        self._alerts_this_hour.append(time.time())

        return True

    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if we're allowed to send this alert."""
        now = time.time()

        # Cooldown per alert type
        last = self._last_alert_by_type.get(alert_type, 0)
        if now - last < self.cooldown:
            return False

        # Hourly limit
        one_hour_ago = now - 3600
        recent = [t for t in self._alerts_this_hour if t > one_hour_ago]
        if len(recent) >= self.max_per_hour:
            return False

        return True

    # ── Channel Implementations ──

    def _send_slack(self, webhook_url: str, title: str, message: str, severity: str) -> None:
        color = {"INFO": "#36a64f", "WARNING": "#ff9900", "CRITICAL": "#ff0000"}.get(severity, "#000000")
        payload = {
            "attachments": [{
                "color": color,
                "title": title,
                "text": message.replace("**", "*"),  # Slack uses single *
                "footer": "microstructure-alpha",
                "ts": int(time.time()),
            }]
        }
        self._post_json(webhook_url, payload)

    def _send_telegram(self, token: str, chat_id: str, message: str) -> None:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        # Convert markdown bold from ** to Telegram format
        tg_message = message.replace("**", "*")
        payload = {
            "chat_id": chat_id,
            "text": tg_message,
            "parse_mode": "Markdown",
        }
        self._post_json(url, payload)

    def _send_discord(self, webhook_url: str, title: str, message: str, severity: str) -> None:
        color = {"INFO": 3066993, "WARNING": 16776960, "CRITICAL": 16711680}.get(severity, 0)
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "footer": {"text": "microstructure-alpha"},
            }]
        }
        self._post_json(webhook_url, payload)

    @staticmethod
    def _post_json(url: str, payload: Dict, timeout: float = 10.0) -> None:
        """POST JSON to a URL."""
        data = json.dumps(payload).encode('utf-8')
        req = urllib_request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            urllib_request.urlopen(req, timeout=timeout)
        except URLError as e:
            raise RuntimeError(f"Webhook POST failed: {e}")

    # ── Query ──

    def get_recent_alerts(self, n: int = 20) -> List[Dict]:
        with self._lock:
            alerts = list(self._history)[-n:]
        return [
            {
                'timestamp': a.timestamp,
                'title': a.title,
                'severity': a.severity,
                'channels': a.channels_sent,
                'success': a.success,
            }
            for a in alerts
        ]

    @property
    def is_configured(self) -> bool:
        return bool(self._slack_webhooks or self._telegram_configs or self._discord_webhooks)
