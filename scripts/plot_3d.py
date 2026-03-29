# fais peter les graphiques
# petits dessins

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_decision_log():
    # on ramene les datas
    path = os.path.join(BASE_DIR, "decision_log.csv")
    if not os.path.exists(path):
        print(f"No decision_log.csv found at {path}")
        return None

    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "regime_id": row.get("regime_id", "-1"),
                    "delta_mu": float(row.get("delta_mu", 0)),
                    "delta_sigma": float(row.get("delta_sigma", 0)),
                    "delta_entropy": float(row.get("delta_entropy", 0)),
                    "delta_skew": float(row.get("delta_skew", 0)),
                    "transition_strength": float(row.get("transition_strength", 0)),
                    "tail_risk": float(row.get("tail_risk", 0)),
                    "l2_liquidity_slope": float(row.get("l2_liquidity_slope", 0)),
                    "result": row.get("result", "FLAT"),
                    "action": row.get("action", "FLAT"),
                })
            except (ValueError, KeyError):
                continue
    return data


def load_trade_journal():
    # on ramene les datas
    path = os.path.join(BASE_DIR, "trade_journal.csv")
    if not os.path.exists(path):
        print(f"No trade_journal.csv found at {path}")
        return None

    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "trade_id": int(row.get("trade_id", 0)),
                    "regime_id": int(row.get("regime_id", -1)),
                    "net_pnl": float(row.get("net_pnl", 0)),
                    "hold_duration_windows": int(row.get("hold_duration_windows", 0)),
                    "entry_price": float(row.get("entry_price", 0)),
                    "exit_price": float(row.get("exit_price", 0)),
                    "stop_loss_pct": float(row.get("stop_loss_pct", 0)),
                    "take_profit_pct": float(row.get("take_profit_pct", 0)),
                    "transition_strength": float(row.get("transition_strength", 0)),
                    "volatility_at_entry": float(row.get("volatility_at_entry", 0)),
                    "regime_confidence": float(row.get("regime_confidence", 0)),
                    "exit_reason": row.get("exit_reason", ""),
                    "side": row.get("side", ""),
                })
            except (ValueError, KeyError):
                continue
    return data


def load_market_state_log():
    # on ramene les datas
    # dans quel etat j'erre
    path = os.path.join(BASE_DIR, "market_state_log.csv")
    if not os.path.exists(path):
        print(f"No market_state_log.csv found at {path}")
        return None

    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "timestamp": float(row.get("timestamp", 0)),
                    "mu": float(row.get("mu", 0)),
                    "sigma": float(row.get("sigma", 0)),
                    "entropy": float(row.get("entropy", 0)),
                    "skew": float(row.get("skew", 0)),
                    "kurtosis": float(row.get("kurtosis", 0)),
                    "regime_id": int(row.get("regime_id", -1))
                })
            except (ValueError, KeyError):
                continue
    return data


def plot_1_state_space(decision_data, save_path):
    # fais peter les graphiques
    # dans quel etat j'erre
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter for non-trivial points
    points = [d for d in decision_data if abs(d["delta_mu"]) > 1e-10 or abs(d["delta_sigma"]) > 1e-10]
    if not points:
        print("  No state-space data to plot.")
        return

    x = np.array([d["delta_mu"] for d in points])
    y = np.array([d["delta_sigma"] for d in points])
    z = np.array([d["delta_entropy"] for d in points])

    # Color by regime
    regimes = [d["regime_id"] for d in points]
    unique_regimes = sorted(set(regimes))
    cmap = cm.get_cmap("tab10", max(1, len(unique_regimes)))
    regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}
    colors = [cmap(regime_to_idx[r]) for r in regimes]

    # Size by transition strength
    sizes = np.clip(np.array([d["transition_strength"] for d in points]) * 100, 5, 80)

    # Mark trades vs non-trades
    is_trade = np.array([d["result"] == "TRADE" for d in points])

    # Plot non-trades as small dots
    if np.any(~is_trade):
        ax.scatter(x[~is_trade], y[~is_trade], z[~is_trade],
                   c=[colors[i] for i in range(len(colors)) if not is_trade[i]],
                   s=sizes[~is_trade] * 0.3, alpha=0.15, edgecolors='none')

    # Plot trades as large markers
    if np.any(is_trade):
        ax.scatter(x[is_trade], y[is_trade], z[is_trade],
                   c=[colors[i] for i in range(len(colors)) if is_trade[i]],
                   s=sizes[is_trade] * 2, alpha=0.9, edgecolors='black',
                   linewidths=0.5, marker='D')

    ax.set_xlabel('Δμ (Drift Change)', fontsize=10)
    ax.set_ylabel('Δσ (Volatility Change)', fontsize=10)
    ax.set_zlabel('ΔH (Entropy Change)', fontsize=10)
    ax.set_title('3D Microstructure State Space\n(Diamonds = Executed Trades)', fontsize=13, fontweight='bold')

    # Legend
    for regime in unique_regimes:
        idx = regime_to_idx[regime]
        ax.scatter([], [], [], c=[cmap(idx)], label=f"Regime {regime}", s=30)
    ax.legend(loc='upper left', fontsize=8)

    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_2_exit_surface(trade_data, save_path):
    # fais peter les graphiques
    # ca secoue
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not trade_data or len(trade_data) < 2:
        print("  Not enough trade data for exit surface.")
        return

    vol = np.array([d["volatility_at_entry"] for d in trade_data])
    strength = np.array([d["transition_strength"] for d in trade_data])
    sl = np.array([d["stop_loss_pct"] for d in trade_data]) * 100  # Convert to %
    tp = np.array([d["take_profit_pct"] for d in trade_data]) * 100
    pnl = np.array([d["net_pnl"] for d in trade_data])

    # Color by PnL (green = win, red = loss)
    pnl_colors = np.where(pnl > 0, 'limegreen', 'crimson')

    # Plot stop-loss levels
    ax.scatter(vol, strength, sl, c='orangered', s=60, alpha=0.8,
               label='Stop-Loss %', marker='v', edgecolors='black', linewidth=0.3)

    # Plot take-profit levels
    ax.scatter(vol, strength, tp, c='dodgerblue', s=60, alpha=0.8,
               label='Take-Profit %', marker='^', edgecolors='black', linewidth=0.3)

    # Connect SL to TP for each trade
    for i in range(len(trade_data)):
        ax.plot([vol[i], vol[i]], [strength[i], strength[i]], [sl[i], tp[i]],
                color=pnl_colors[i], alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Entry Volatility (σ)', fontsize=10)
    ax.set_ylabel('Transition Strength', fontsize=10)
    ax.set_zlabel('Exit Level (%)', fontsize=10)
    ax.set_title('3D Dynamic Exit Levels\n(Red bars = Losses, Green bars = Wins)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_3_pnl_landscape(trade_data, save_path):
    # fais peter les graphiques
    # ca fait bim
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not trade_data or len(trade_data) < 2:
        print("  Not enough trade data for PnL landscape.")
        return

    regimes = np.array([d["regime_id"] for d in trade_data])
    holds = np.array([d["hold_duration_windows"] for d in trade_data])
    pnls = np.array([d["net_pnl"] for d in trade_data])
    confidences = np.array([d["regime_confidence"] for d in trade_data])

    # Color by PnL
    colors = []
    for p in pnls:
        if p > 0:
            intensity = min(1.0, abs(p) / max(1.0, max(abs(pnls))))
            colors.append((0.2, 0.8, 0.2, 0.5 + 0.5 * intensity))
        else:
            intensity = min(1.0, abs(p) / max(1.0, max(abs(pnls))))
            colors.append((0.9, 0.2, 0.2, 0.5 + 0.5 * intensity))

    # Bar width
    dx = dy = 0.4
    dz = pnls

    ax.bar3d(regimes, holds, np.zeros_like(pnls), dx, dy * 2, dz,
             color=colors, edgecolor='black', linewidth=0.3, alpha=0.85)

    # Overlay confidence as text
    for i in range(len(trade_data)):
        ax.text(regimes[i], holds[i], pnls[i],
                f'  c={confidences[i]:.2f}', fontsize=6, color='navy')

    ax.set_xlabel('Regime ID', fontsize=10)
    ax.set_ylabel('Hold Duration (windows)', fontsize=10)
    ax.set_zlabel('Net PnL ($)', fontsize=10)
    ax.set_title('3D PnL Landscape by Regime & Hold Duration\n(Height = PnL, Color = Win/Loss)',
                 fontsize=13, fontweight='bold')

    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_4_signal_quality(trade_data, save_path):
    # fais peter les graphiques
    # le feu vert
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not trade_data or len(trade_data) < 2:
        print("  Not enough trade data for signal quality plot.")
        return

    strength = np.array([d["transition_strength"] for d in trade_data])
    confidence = np.array([d["regime_confidence"] for d in trade_data])
    volatility = np.array([d["volatility_at_entry"] for d in trade_data])
    pnls = np.array([d["net_pnl"] for d in trade_data])

    # Color by PnL severity
    colors = []
    max_pnl = max(max(abs(pnls)), 1e-9)
    for p in pnls:
        intensity = min(1.0, abs(p) / max_pnl)
        if p > 0:
            colors.append((0.0, 0.8, 0.0, 0.4 + 0.6 * intensity))
        else:
            colors.append((0.8, 0.0, 0.0, 0.4 + 0.6 * intensity))

    sizes = np.clip(np.abs(pnls) / max_pnl * 300, 20, 300)

    ax.scatter(strength, confidence, volatility, c=colors, s=sizes, edgecolors='black', alpha=0.8, linewidth=0.5)

    ax.set_xlabel('Transition Strength', fontsize=10)
    ax.set_ylabel('Regime Confidence', fontsize=10)
    ax.set_zlabel('Entry Volatility (σ)', fontsize=10)
    ax.set_title('3D Signal Quality & Outcomes\n(Size = PnL Magnitude, Green = Profit, Red = Loss)', fontsize=13, fontweight='bold')

    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_5_opportunity_landscape(decision_data, save_path):
    # attention aux degats
    # fais peter les graphiques
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter for interesting entries
    points = [d for d in decision_data if abs(d["transition_strength"]) > 1e-10 or d["result"] in ["TRADE", "REJECTED"]]
    if not points:
        print("  Not enough decision data for opportunity landscape.")
        return

    liq = np.array([d["l2_liquidity_slope"] for d in points])
    strength = np.array([d["transition_strength"] for d in points])
    tail = np.array([d["tail_risk"] for d in points])
    results = np.array([d["result"] for d in points])

    # Plot each category
    trade_mask = results == "TRADE"
    rejected_mask = results == "REJECTED"
    flat_mask = ~(trade_mask | rejected_mask)

    if np.any(flat_mask):
        ax.scatter(liq[flat_mask], strength[flat_mask], tail[flat_mask],
                   c='gray', s=10, alpha=0.1, label='Flat', edgecolors='none')

    if np.any(rejected_mask):
        ax.scatter(liq[rejected_mask], strength[rejected_mask], tail[rejected_mask],
                   c='orange', s=30, alpha=0.4, label='Rejected Signal', edgecolors='none', marker='x')

    if np.any(trade_mask):
        ax.scatter(liq[trade_mask], strength[trade_mask], tail[trade_mask],
                   c='blue', s=80, alpha=0.9, label='Executed Trade', edgecolors='black', marker='*')

    ax.set_xlabel('L2 Liquidity Slope', fontsize=10)
    ax.set_ylabel('Transition Strength', fontsize=10)
    ax.set_zlabel('Tail Risk', fontsize=10)
    ax.set_title('3D Opportunity Landscape\nEvaluating All Signals Generated by the Engine', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    ax.view_init(elev=30, azim=60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_6_volatility_drift_surface(state_data, save_path):
    # fais peter les graphiques
    # ca secoue
    import matplotlib.tri as mtri
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not state_data or len(state_data) < 5:
        print("  Not enough state data for surface plot.")
        return

    # To avoid rendering thousands of points on a surface, sample them
    step = max(1, len(state_data) // 500)
    sampled = state_data[::step]
    
    mu = np.array([d["mu"] for d in sampled])
    sigma = np.array([d["sigma"] for d in sampled])
    time_idx = np.arange(len(sampled))
    
    # Create surface mesh
    try:
        tri = mtri.Triangulation(mu, sigma)
        surf = ax.plot_trisurf(mu, sigma, time_idx, triangles=tri.triangles, cmap='plasma', alpha=0.8, edgecolor='none')
        fig.colorbar(surf, ax=ax, label='Time Progression (Windows)', pad=0.1)
    except Exception as e:
        # Fallback to scatter if triangulation fails
        scatter = ax.scatter(mu, sigma, time_idx, c=time_idx, cmap='plasma', s=20)
        fig.colorbar(scatter, ax=ax, label='Time Progression (Windows)', pad=0.1)

    # Add trajectory line to track the shifts clearly
    ax.plot(mu, sigma, time_idx, color='black', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Drift (μ)', fontsize=10)
    ax.set_ylabel('Volatility (σ)', fontsize=10)
    ax.set_zlabel('Time Index', fontsize=10)
    ax.set_title('3D Regime Shift Surface\n(Tracking Volatility vs Drift Over Time)', fontsize=13, fontweight='bold')

    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_7_density_evolution(state_data, save_path):
    # fais peter les graphiques
    from scipy.stats import t, norm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection # Moved import here
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not state_data or len(state_data) < 5:
        print("  Not enough state data for density evolution plot.")
        return

    # Sample windows
    num_slices = min(60, len(state_data))
    step = max(1, len(state_data) // num_slices)
    sampled = state_data[::step]

    # Global x range based on global min/max
    all_mu = np.array([d["mu"] for d in sampled])
    all_sigma = np.array([d["sigma"] for d in sampled])
    x_min = np.min(all_mu - 4 * all_sigma)
    x_max = np.max(all_mu + 4 * all_sigma)
    x = np.linspace(x_min, x_max, 300)

    for i, d in enumerate(sampled):
        y_time = i
        mu = d["mu"]
        sigma = max(1e-8, d["sigma"])
        kurt = max(0, d["kurtosis"] - 3) # excess kurtosis
        
        # Approximate Fat tails using Student's t distribution
        if kurt > 0.1:
            df = max(2.1, 6 / kurt + 4)
            scale = sigma * np.sqrt((df - 2) / df)
            z = t.pdf(x, df, loc=mu, scale=scale)
        else:
            z = norm.pdf(x, loc=mu, scale=sigma)
            
        # Add small offset for visibility
        z = z + 1e-6
        
        # Fill polygon (Waterfall slice)
        ax.plot(x, np.full_like(x, y_time), z, color='midnightblue', linewidth=0.8)
        
        # Color by kurtosis (fat tails)
        color = cm.Reds(min(1.0, kurt / 15.0))
        # Need to format collection explicitly for 3d fill
        verts = [(x_min, 0)] + list(zip(x, z)) + [(x_max, 0)]
        poly = Poly3DCollection([list(zip(x, np.full_like(x, y_time), z))], facecolors=color, alpha=0.6)
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, len(sampled))
    
    # Hide Z axis scale mostly, just relative intensity
    ax.set_zticks([])

    ax.set_xlabel('Returns Space', fontsize=10)
    ax.set_ylabel('Time (Evolution)', fontsize=10)
    ax.set_zlabel('Probability Density', fontsize=10)
    ax.set_title('3D Probability Density Evolution\n(High Kurtosis / Fat Tails Highlighted in Red)', fontsize=13, fontweight='bold')

    # View down the time axis to see shifts and overlap
    ax.view_init(elev=35, azim=-45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("=" * 60)
    print("  3D REGIME & EXIT VISUALIZATION")
    print("=" * 60)

    decision_data = load_decision_log()
    trade_data = load_trade_journal()
    state_data = load_market_state_log()

    # Plot 1: State Space
    if decision_data:
        print(f"\n  Loaded {len(decision_data)} decision log entries.")
        plot_1_state_space(decision_data, os.path.join(BASE_DIR, "3d_state_space.png"))
    else:
        print("  Skipping state-space plot (no decision_log.csv)")

    # Plot 2: Exit Surfaces
    if trade_data:
        print(f"  Loaded {len(trade_data)} trade journal entries.")
        plot_2_exit_surface(trade_data, os.path.join(BASE_DIR, "3d_exit_surface.png"))
        plot_3_pnl_landscape(trade_data, os.path.join(BASE_DIR, "3d_pnl_landscape.png"))
        plot_4_signal_quality(trade_data, os.path.join(BASE_DIR, "3d_signal_quality.png"))
    else:
        print("  Skipping exit/PnL/signal plots (no trade_journal.csv)")

    # Plot 5: Opportunity Landscape
    if decision_data:
        plot_5_opportunity_landscape(decision_data, os.path.join(BASE_DIR, "3d_opportunity_landscape.png"))

    # Plot 6 & 7: Dynamics
    if state_data:
        print(f"  Loaded {len(state_data)} market state log entries.")
        plot_6_volatility_drift_surface(state_data, os.path.join(BASE_DIR, "3d_volatility_drift_surface.png"))
        plot_7_density_evolution(state_data, os.path.join(BASE_DIR, "3d_density_evolution.png"))
    else:
        print("  Skipping surface/density plots (no market_state_log.csv)")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
