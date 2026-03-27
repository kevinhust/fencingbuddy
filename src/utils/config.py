"""
FencerAI Configuration Management
==================================
Version: 1.0 | Last Updated: 2026-03-27

OmegaConf-style configuration management with YAML/CLI override support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


# =============================================================================
# Configuration Schema
# =============================================================================

@dataclass
class ModelPaths:
    """Paths to model files."""
    rtmpose_model: str = "models/rtmpose-l.onnx"
    pose_embedder_model: str = "models/pose_embedder.onnx"


@dataclass
class TrackerParams:
    """Norfair tracker parameters."""
    max_distance: float = 30.0
    max_age: int = 30
    min_hits: int = 3


@dataclass
class HomographySettings:
    """Homography calibration settings."""
    piste_length: float = 14.0  # meters
    piste_width: float = 2.0  # meters


@dataclass
class EmaAlphas:
    """EMA smoothing alpha values."""
    velocity_alpha: float = 0.7
    acceleration_alpha: float = 0.7


@dataclass
class FeatureFlags:
    """Feature extraction toggles."""
    extract_static_geometry: bool = True
    extract_distance: bool = True
    extract_angular: bool = True
    extract_velocity: bool = True
    extract_acceleration: bool = True


@dataclass
class Config:
    """FencerAI configuration schema."""
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    tracker_params: TrackerParams = field(default_factory=TrackerParams)
    homography_settings: HomographySettings = field(default_factory=HomographySettings)
    ema_alphas: EmaAlphas = field(default_factory=EmaAlphas)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = Config()


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dictionary of values to override

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If config file does not exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_overrides:
        data = _deep_merge(data, cli_overrides)

    return merge_config(data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config instance to save
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    data = {
        'model_paths': _dataclass_to_dict(config.model_paths),
        'tracker_params': _dataclass_to_dict(config.tracker_params),
        'homography_settings': _dataclass_to_dict(config.homography_settings),
        'ema_alphas': _dataclass_to_dict(config.ema_alphas),
        'feature_flags': _dataclass_to_dict(config.feature_flags),
    }
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def merge_config(partial: Dict[str, Any]) -> Config:
    """
    Merge partial config dict with defaults.

    Args:
        partial: Partial config dictionary

    Returns:
        Merged Config instance
    """
    default_dict = _config_to_dict(DEFAULT_CONFIG)
    merged = _deep_merge(default_dict, partial)
    return _dict_to_config(merged)


# =============================================================================
# Helper Functions
# =============================================================================

def _config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Convert Config dataclass to nested dict."""
    if hasattr(cfg, '__dataclass_fields__'):
        result = {}
        for key in cfg.__dataclass_fields__:
            value = getattr(cfg, key)
            if hasattr(value, '__dataclass_fields__'):
                result[key] = _config_to_dict(value)
            else:
                result[key] = value
        return result
    return cfg


def _dataclass_to_dict(dc) -> Dict[str, Any]:
    """Convert any dataclass to dict."""
    result = {}
    for key in dc.__dataclass_fields__:
        value = getattr(dc, key)
        if hasattr(value, '__dataclass_fields__'):
            result[key] = _dataclass_to_dict(value)
        else:
            result[key] = value
    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert nested dict to Config instance."""
    return Config(
        model_paths=ModelPaths(**data.get('model_paths', {})),
        tracker_params=TrackerParams(**data.get('tracker_params', {})),
        homography_settings=HomographySettings(**data.get('homography_settings', {})),
        ema_alphas=EmaAlphas(**data.get('ema_alphas', {})),
        feature_flags=FeatureFlags(**data.get('feature_flags', {})),
    )
