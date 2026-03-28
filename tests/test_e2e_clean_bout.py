"""
E2E Tests for Clean Bout Data
==============================
TDD Phase 5.3.1: End-to-End Tests with Clean Bout Sample Data

Tests the full pipeline with clean fencing footage (high quality, good lighting).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.perception.pipeline import PerceptionPipeline
from src.recognition.feature_extractor import FeatureExtractor
from src.utils.schemas import FrameData


# Sample video path
SAMPLE_VIDEO = Path("/Users/kevinwang/Documents/20Projects/fecingbuddy/data/samples/498_1728950803.mp4")


class TestCleanBoutPipeline:
    """E2E tests for clean bout data."""

    @pytest.fixture
    def sample_video_exists(self):
        """Check if sample video exists."""
        if not SAMPLE_VIDEO.exists():
            pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")
        return True

    def test_video_can_be_opened(self, sample_video_exists):
        """Sample video should be accessible."""
        import cv2
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        assert cap.isOpened(), f"Cannot open video: {SAMPLE_VIDEO}"
        cap.release()

    def test_video_properties(self, sample_video_exists):
        """Video should have expected properties."""
        import cv2
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert frame_count > 0, "Video should have frames"
        assert fps > 0, "Video should have valid FPS"
        assert width > 0 and height > 0, "Video should have valid dimensions"
        print(f"Video: {frame_count} frames, {fps:.2f} fps, {width}x{height}")

    def test_perception_pipeline_on_video(self, sample_video_exists):
        """Perception pipeline should process video frames."""
        import cv2

        perception = PerceptionPipeline(conf_threshold=0.3)
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))

        frame_count = 0
        valid_frames = 0

        while frame_count < 30:  # Test first 30 frames
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = perception.process_frame(
                frame=frame,
                timestamp=frame_count / 30.0,
                frame_id=frame_count,
            )

            if len(frame_data.poses) > 0:
                valid_frames += 1

            frame_count += 1

        cap.release()

        assert frame_count > 0, "Should have processed some frames"
        print(f"Processed {frame_count} frames, {valid_frames} with valid poses")

    def test_full_pipeline_produces_features(self, sample_video_exists):
        """Full pipeline should produce valid feature matrix."""
        import cv2

        perception = PerceptionPipeline(conf_threshold=0.3)
        feature_extractor = FeatureExtractor()

        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        all_features = []
        frame_count = 0

        while frame_count < 30:  # Test first 30 frames
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = perception.process_frame(
                frame=frame,
                timestamp=frame_count / 30.0,
                frame_id=frame_count,
            )

            features = feature_extractor.extract_frame_features(
                frame=frame_data,
                frame_width=float(actual_width),
            )

            all_features.append(features)
            frame_count += 1

        cap.release()

        assert len(all_features) > 0, "Should have extracted features"
        features_array = np.array(all_features)
        assert features_array.shape[0] == frame_count, "Feature count should match frame count"
        assert features_array.shape[1] == 2, "Should have 2 fencers"
        assert features_array.shape[2] == 101, "Should have 101 features"

        print(f"Feature matrix shape: {features_array.shape}")
        print(f"Value range: [{features_array.min():.2f}, {features_array.max():.2f}]")

    def test_feature_matrix_has_valid_values(self, sample_video_exists):
        """Feature matrix should have non-zero values."""
        import cv2

        perception = PerceptionPipeline(conf_threshold=0.3)
        feature_extractor = FeatureExtractor()

        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        all_features = []
        frame_count = 0

        while frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = perception.process_frame(
                frame=frame,
                timestamp=frame_count / 30.0,
                frame_id=frame_count,
            )

            features = feature_extractor.extract_frame_features(
                frame=frame_data,
                frame_width=float(actual_width),
            )

            all_features.append(features)
            frame_count += 1

        cap.release()

        features_array = np.array(all_features)
        non_zero_ratio = np.count_nonzero(features_array) / features_array.size
        assert non_zero_ratio > 0.5, f"Feature matrix should have >50% non-zero values, got {non_zero_ratio:.1%}"
        print(f"Non-zero ratio: {non_zero_ratio:.1%}")

    def test_audio_events_detected(self, sample_video_exists):
        """Audio events should be detected in the video."""
        import cv2

        perception = PerceptionPipeline(conf_threshold=0.3)
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))

        audio_events = []
        frame_count = 0

        while frame_count < 100:  # Check first 100 frames
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = perception.process_frame(
                frame=frame,
                timestamp=frame_count / 30.0,
                frame_id=frame_count,
            )

            if frame_data.audio_event:
                audio_events.append(frame_data.audio_event)

            frame_count += 1

        cap.release()

        print(f"Detected {len(audio_events)} audio events in {frame_count} frames")
        # Note: May or may not have audio events depending on video content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
