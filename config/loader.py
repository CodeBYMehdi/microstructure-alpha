import yaml
import os
from pathlib import Path
from typing import Dict, Any
from .schema import AppConfig, ExecutionConfig, InstrumentsConfig, ThresholdsConfig

class ConfigLoader:
    def __init__(self, config_dir: str = None, profile: str = None):
        if config_dir is None:
            # Défaut rép courant si non fourni
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = Path(config_dir)
        self.profile = profile  # e.g. "nasdaq" loads execution_live.yaml, instruments_nasdaq.yaml, thresholds_nasdaq.yaml
        self._config: AppConfig = None

    def load(self, profile: str = None) -> AppConfig:
        # on ramene les datas
        # les petits reglages
        p = profile or self.profile
        if p:
            exec_file = f"execution_{p}.yaml" if (self.config_dir / f"execution_{p}.yaml").exists() else "execution_live.yaml"
            instr_file = f"instruments_{p}.yaml" if (self.config_dir / f"instruments_{p}.yaml").exists() else "instruments.yaml"
            thresh_file = f"thresholds_{p}.yaml" if (self.config_dir / f"thresholds_{p}.yaml").exists() else "thresholds.yaml"
        else:
            exec_file = "execution.yaml"
            instr_file = "instruments.yaml"
            thresh_file = "thresholds.yaml"

        execution_data = self._load_yaml(exec_file)
        instruments_data = self._load_yaml(instr_file)
        thresholds_data = self._load_yaml(thresh_file)

        # Construit objet config complet
        # Note: Structure yaml doit peut-être être ajustée pour matcher AppConfig
        # ou on map ici.
        
        # execution.yaml a clé racine 'execution'
        execution_config = ExecutionConfig(**execution_data['execution'])
        
        # instruments.yaml a clé racine 'instruments'
        instruments_config = InstrumentsConfig(instruments=instruments_data['instruments'])
        
        # thresholds.yaml a clés 'regime', 'risk', 'pdf' à la racine
        thresholds_config = ThresholdsConfig(
            regime=thresholds_data['regime'],
            risk=thresholds_data['risk'],
            pdf=thresholds_data['pdf'],
            decision=thresholds_data['decision']
        )

        self._config = AppConfig(
            execution=execution_config,
            instruments=instruments_config,
            thresholds=thresholds_config
        )
        return self._config

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier config non trouvé: {filepath}")
        
        with open(filepath, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Err parsing YAML {filename}: {e}")

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            self.load()
        return self._config

# instance globale
_loader = ConfigLoader()

def get_config() -> AppConfig:
    return _loader.config

def reload_config():
    _loader.load()

def load_profile(profile: str) -> AppConfig:
    """Load a named config profile (e.g. 'nasdaq' for IBKR paper trading)."""
    global _loader
    _loader = ConfigLoader(profile=profile)
    return _loader.load(profile)
