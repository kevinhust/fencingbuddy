"""
Tests for src/utils/config.py
TDD Phase 1.1: Config Management
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile
import os


class TestConfigSchema:
    """Test configuration schema validation."""

    def test_default_config_has_required_fields(self):
        """Config should have model_paths, tracker_params, homography_settings, ema_alphas, feature_flags."""
        from src.utils.config import Config, DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
        assert hasattr(cfg, 'model_paths')
        assert hasattr(cfg, 'tracker_params')
        assert hasattr(cfg, 'homography_settings')
        assert hasattr(cfg, 'ema_alphas')
        assert hasattr(cfg, 'feature_flags')

    def test_model_paths_has_required_keys(self):
        """model_paths should contain rtmpose_model, pose_embedder_model."""
        from src.utils.config import DEFAULT_CONFIG
        model_paths = DEFAULT_CONFIG.model_paths
        assert hasattr(model_paths, 'rtmpose_model')
        assert hasattr(model_paths, 'pose_embedder_model')

    def test_tracker_params_has_required_keys(self):
        """tracker_params should contain max_distance, max_age, min_hits."""
        from src.utils.config import DEFAULT_CONFIG
        tracker_params = DEFAULT_CONFIG.tracker_params
        assert hasattr(tracker_params, 'max_distance')
        assert hasattr(tracker_params, 'max_age')
        assert hasattr(tracker_params, 'min_hits')

    def test_homography_settings_has_required_keys(self):
        """homography_settings should contain piste_length, piste_width."""
        from src.utils.config import DEFAULT_CONFIG
        homography = DEFAULT_CONFIG.homography_settings
        assert hasattr(homography, 'piste_length')
        assert hasattr(homography, 'piste_width')

    def test_ema_alphas_has_required_keys(self):
        """ema_alphas should contain velocity_alpha, acceleration_alpha."""
        from src.utils.config import DEFAULT_CONFIG
        ema = DEFAULT_CONFIG.ema_alphas
        assert hasattr(ema, 'velocity_alpha')
        assert hasattr(ema, 'acceleration_alpha')
        # Validate alpha range
        assert 0.0 <= ema.velocity_alpha <= 1.0
        assert 0.0 <= ema.acceleration_alpha <= 1.0

    def test_feature_flags_has_required_keys(self):
        """feature_flags should control extraction."""
        from src.utils.config import DEFAULT_CONFIG
        flags = DEFAULT_CONFIG.feature_flags
        assert hasattr(flags, 'extract_static_geometry')
        assert hasattr(flags, 'extract_distance')
        assert hasattr(flags, 'extract_angular')
        assert hasattr(flags, 'extract_velocity')
        assert hasattr(flags, 'extract_acceleration')


class TestConfigLoad:
    """Test configuration loading from YAML."""

    def test_load_from_yaml_file(self):
        """Should load config from YAML file."""
        from src.utils.config import Config, load_config
        import yaml

        # Use explicit file open/close to ensure data is flushed
        temp_path = '/tmp/test_load_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump({'model_paths': {'rtmpose_model': '/test/model.onnx'}}, f)

        try:
            cfg = load_config(temp_path)
            assert cfg.model_paths.rtmpose_model == '/test/model.onnx'
        finally:
            os.unlink(temp_path)

    def test_load_from_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for nonexistent file."""
        from src.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')

    def test_load_with_cli_overrides(self):
        """Should override config values from CLI."""
        from src.utils.config import load_config
        import yaml

        temp_path = '/tmp/test_load_overrides.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump({'model_paths': {'rtmpose_model': '/original/path'}}, f)

        try:
            overrides = {'model_paths': {'rtmpose_model': '/overridden/path'}}
            cfg = load_config(temp_path, cli_overrides=overrides)
            assert cfg.model_paths.rtmpose_model == '/overridden/path'
        finally:
            os.unlink(temp_path)


class TestConfigSave:
    """Test configuration saving."""

    def test_save_to_yaml(self):
        """Should save config to YAML file."""
        from src.utils.config import DEFAULT_CONFIG, save_config
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_config(DEFAULT_CONFIG, temp_path)
            with open(temp_path) as f:
                loaded = yaml.safe_load(f)
            assert 'model_paths' in loaded
            assert 'tracker_params' in loaded
        finally:
            os.unlink(temp_path)


class TestConfigMerge:
    """Test configuration merging."""

    def test_merge_partial_config(self):
        """Should merge partial config with defaults."""
        from src.utils.config import merge_config, DEFAULT_CONFIG
        partial = {'model_paths': {'rtmpose_model': '/new/path'}}
        merged = merge_config(partial)
        assert merged.model_paths.rtmpose_model == '/new/path'
        # Other values should be from defaults
        assert merged.model_paths.pose_embedder_model == DEFAULT_CONFIG.model_paths.pose_embedder_model

    def test_merge_empty_config_returns_defaults(self):
        """Empty config should return defaults."""
        from src.utils.config import merge_config, DEFAULT_CONFIG
        merged = merge_config({})
        assert merged.model_paths.rtmpose_model == DEFAULT_CONFIG.model_paths.rtmpose_model
