"""
Tests for src/recognition/feature_extractor.py canonicalization functions
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fecingbuddy')

from src.recognition.feature_extractor import (
    canonicalize_pose,
    canonicalize_frame,
)
from src.utils.schemas import FencerPose, Keypoint, FrameData


class TestCanonicalizePose:
    """Test pose canonicalization."""

    def test_left_fencer_unchanged(self):
        """Left fencer (id=0) should not be flipped."""
        pose = FencerPose(
            fencer_id=0,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),   # nose
                Keypoint(x=120, y=40, conf=0.9),  # left_eye
                Keypoint(x=130, y=40, conf=0.9), # right_eye
                Keypoint(x=110, y=45, conf=0.9),  # left_ear
                Keypoint(x=140, y=45, conf=0.9), # right_ear
                Keypoint(x=150, y=100, conf=0.9), # left_shoulder
                Keypoint(x=250, y=100, conf=0.9), # right_shoulder
                Keypoint(x=140, y=150, conf=0.9), # left_elbow
                Keypoint(x=260, y=150, conf=0.9), # right_elbow
                Keypoint(x=130, y=200, conf=0.9), # left_wrist
                Keypoint(x=270, y=200, conf=0.9), # right_wrist
                Keypoint(x=170, y=250, conf=0.9), # left_hip
                Keypoint(x=230, y=250, conf=0.9), # right_hip
                Keypoint(x=165, y=350, conf=0.9), # left_knee
                Keypoint(x=235, y=350, conf=0.9), # right_knee
                Keypoint(x=160, y=450, conf=0.9), # left_ankle
                Keypoint(x=240, y=450, conf=0.9), # right_ankle
            ],
            is_canonical_flipped=False,
        )

        result = canonicalize_pose(pose)
        assert result.fencer_id == 0
        assert result.is_canonical_flipped == False

    def test_right_fencer_flips_horizontally(self):
        """Right fencer (id=1) should be flipped horizontally."""
        frame_width = 1920.0
        pose = FencerPose(
            fencer_id=1,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),   # nose
                Keypoint(x=120, y=40, conf=0.9),  # left_eye
                Keypoint(x=130, y=40, conf=0.9), # right_eye
                Keypoint(x=110, y=45, conf=0.9),  # left_ear
                Keypoint(x=140, y=45, conf=0.9), # right_ear
                Keypoint(x=150, y=100, conf=0.9), # left_shoulder
                Keypoint(x=250, y=100, conf=0.9), # right_shoulder
                Keypoint(x=140, y=150, conf=0.9), # left_elbow
                Keypoint(x=260, y=150, conf=0.9), # right_elbow
                Keypoint(x=130, y=200, conf=0.9), # left_wrist
                Keypoint(x=270, y=200, conf=0.9), # right_wrist
                Keypoint(x=170, y=250, conf=0.9), # left_hip
                Keypoint(x=230, y=250, conf=0.9), # right_hip
                Keypoint(x=165, y=350, conf=0.9), # left_knee
                Keypoint(x=235, y=350, conf=0.9), # right_knee
                Keypoint(x=160, y=450, conf=0.9), # left_ankle
                Keypoint(x=240, y=450, conf=0.9), # right_ankle
            ],
            is_canonical_flipped=False,
        )

        result = canonicalize_pose(pose)

        # After flipping, x = frame_width - original_x
        assert result.fencer_id == 1
        assert result.is_canonical_flipped == True
        # Original nose at x=100 should become x=1820
        assert result.keypoints[0].x == pytest.approx(1820.0, abs=0.1)
        # Original right_shoulder at x=250 should become x=1670
        assert result.keypoints[6].x == pytest.approx(1670.0, abs=0.1)

    def test_already_canonical_right_fencer_unchanged(self):
        """Already flipped right fencer should not be flipped again."""
        pose = FencerPose(
            fencer_id=1,
            bbox=(1720, 50, 1820, 300),
            keypoints=[
                Keypoint(x=1820, y=50, conf=0.9),   # nose
                Keypoint(x=1800, y=40, conf=0.9),  # left_eye
                Keypoint(x=1790, y=40, conf=0.9), # right_eye
                Keypoint(x=1810, y=45, conf=0.9),  # left_ear
                Keypoint(x=1780, y=45, conf=0.9), # right_ear
                Keypoint(x=1770, y=100, conf=0.9), # left_shoulder
                Keypoint(x=1670, y=100, conf=0.9), # right_shoulder
                Keypoint(x=1780, y=150, conf=0.9), # left_elbow
                Keypoint(x=1660, y=150, conf=0.9), # right_elbow
                Keypoint(x=1790, y=200, conf=0.9), # left_wrist
                Keypoint(x=1650, y=200, conf=0.9), # right_wrist
                Keypoint(x=1750, y=250, conf=0.9), # left_hip
                Keypoint(x=1690, y=250, conf=0.9), # right_hip
                Keypoint(x=1755, y=350, conf=0.9), # left_knee
                Keypoint(x=1685, y=350, conf=0.9), # right_knee
                Keypoint(x=1760, y=450, conf=0.9), # left_ankle
                Keypoint(x=1680, y=450, conf=0.9), # right_ankle
            ],
            is_canonical_flipped=True,
        )

        result = canonicalize_pose(pose)
        assert result.is_canonical_flipped == True


class TestCanonicalizeFrame:
    """Test frame canonicalization."""

    def test_empty_frame_unchanged(self):
        """Empty frame should be returned as-is."""
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[],
        )

        result = canonicalize_frame(frame)
        assert len(result.poses) == 0

    def test_single_left_fencer_unchanged(self):
        """Single left fencer should not be flipped."""
        pose = FencerPose(
            fencer_id=0,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
        )

        result = canonicalize_frame(frame)
        assert len(result.poses) == 1
        assert result.poses[0].fencer_id == 0
        assert result.poses[0].is_canonical_flipped == False

    def test_two_fencers_right_flipped(self):
        """Right fencer (id=1) should be flipped, left should not."""
        left_pose = FencerPose(
            fencer_id=0,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )
        right_pose = FencerPose(
            fencer_id=1,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )
        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[left_pose, right_pose],
        )

        result = canonicalize_frame(frame, frame_width=1920.0)

        # Left fencer should still be id=0 and not flipped
        left_result = [p for p in result.poses if p.fencer_id == 0][0]
        assert left_result.is_canonical_flipped == False

        # Right fencer should now be flipped
        right_result = [p for p in result.poses if p.fencer_id == 1][0]
        assert right_result.is_canonical_flipped == True

    def test_frame_preserves_audio_and_homography(self):
        """Canonicalization should preserve audio_event and homography_matrix."""
        pose = FencerPose(
            fencer_id=0,
            bbox=(100, 50, 200, 300),
            keypoints=[
                Keypoint(x=100, y=50, conf=0.9),
                Keypoint(x=120, y=40, conf=0.9),
                Keypoint(x=130, y=40, conf=0.9),
                Keypoint(x=110, y=45, conf=0.9),
                Keypoint(x=140, y=45, conf=0.9),
                Keypoint(x=150, y=100, conf=0.9),
                Keypoint(x=250, y=100, conf=0.9),
                Keypoint(x=140, y=150, conf=0.9),
                Keypoint(x=260, y=150, conf=0.9),
                Keypoint(x=130, y=200, conf=0.9),
                Keypoint(x=270, y=200, conf=0.9),
                Keypoint(x=170, y=250, conf=0.9),
                Keypoint(x=230, y=250, conf=0.9),
                Keypoint(x=165, y=350, conf=0.9),
                Keypoint(x=235, y=350, conf=0.9),
                Keypoint(x=160, y=450, conf=0.9),
                Keypoint(x=240, y=450, conf=0.9),
            ],
            is_canonical_flipped=False,
        )

        from src.utils.schemas import AudioEvent
        audio_event = AudioEvent(timestamp=0.0, event_type="blade_touch", confidence=0.9)
        homography = np.eye(3, dtype=np.float32)

        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            poses=[pose],
            audio_event=audio_event,
            homography_matrix=homography,
        )

        result = canonicalize_frame(frame)
        assert result.audio_event == audio_event
        assert np.array_equal(result.homography_matrix, homography)
