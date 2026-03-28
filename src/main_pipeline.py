"""
FencerAI Main Pipeline
=====================
Version: 1.0 | Last Updated: 2026-03-27

End-to-end pipeline for fencing video feature extraction.
Processes video frames through Perception → Recognition layers to produce (N, 2, 101) feature matrix.

Usage:
    python -m src.main_pipeline --video data/samples/sample.mp4 --output outputs/features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logging import logger
from src.utils.schemas import FeatureMatrix, FrameData
from src.utils.profiling import PipelineMonitor, get_system_info, HealthMonitor
from src.utils.visualization import draw_frame_overlay, draw_info_overlay, export_feature_heatmap
from src.perception.pipeline import PerceptionPipeline
from src.recognition.feature_extractor import FeatureExtractor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FencerAI: Extract 101-dimensional features from fencing videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file (without extension)",
    )
    parser.add_argument(
        "--frame-width",
        type=float,
        default=1920.0,
        help="Frame width for canonicalization",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence for pose detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for pose estimation",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between processed frames (0 = process all)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization output (P1)",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Export feature matrix heatmap (P1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to calibration JSON file",
    )

    return parser.parse_args()


def load_calibration(calibration_path: str) -> Optional[dict]:
    """Load homography calibration from JSON file."""
    if calibration_path is None:
        return None

    calibration_file = Path(calibration_path)
    if not calibration_file.exists():
        logger.warning(f"Calibration file not found: {calibration_path}")
        return None

    with open(calibration_file, "r") as f:
        calibration_data = json.load(f)

    logger.info(f"Loaded calibration from {calibration_path}")
    return calibration_data


def save_features(
    output_path: str,
    feature_matrix: FeatureMatrix,
    metadata: dict,
) -> None:
    """
    Save feature matrix and metadata.

    Args:
        output_path: Output path without extension
        feature_matrix: FeatureMatrix to save
        metadata: Additional metadata dict
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save .npy feature file
    npy_path = f"{output_path}.npy"
    np.save(npy_path, feature_matrix.features)
    logger.info(f"Saved features to {npy_path}")

    # Save .json metadata
    metadata_path = f"{output_path}.json"
    metadata["feature_matrix_shape"] = feature_matrix.features.shape
    metadata["n_frames"] = len(feature_matrix.timestamps)
    metadata["n_fencers"] = 2

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def process_video(
    video_path: str,
    output_path: str,
    frame_width: float = 1920.0,
    confidence_threshold: float = 0.3,
    device: str = "cpu",
    skip_frames: int = 0,
    enable_visualization: bool = False,
    calibration_file: Optional[str] = None,
    enable_profiling: bool = False,
    enable_heatmap: bool = False,
) -> FeatureMatrix:
    """
    Process video through perception and recognition pipelines.

    Args:
        video_path: Path to input video
        output_path: Path to output file (without extension)
        frame_width: Frame width for canonicalization
        confidence_threshold: Minimum confidence for pose detection
        device: Device for pose estimation (cpu/cuda/mps)
        skip_frames: Number of frames to skip between processed frames
        enable_visualization: Enable visualization output
        calibration_file: Optional path to calibration JSON
        enable_profiling: Enable performance profiling

    Returns:
        FeatureMatrix with shape (N_frames, 2, 101)
    """
    # Initialize profiler
    monitor = PipelineMonitor() if enable_profiling else None
    if enable_profiling:
        monitor.start()
        logger.info("Performance profiling enabled")

    # Initialize health monitor
    health_monitor = HealthMonitor(
        min_detections=2,
        min_confidence=0.3,
        max_processing_time_ms=500.0,
    )

    # Load calibration if provided
    calibration_data = load_calibration(calibration_file)

    # Initialize pipelines
    logger.info("Initializing pipelines...")
    perception_pipeline = PerceptionPipeline(
        conf_threshold=confidence_threshold,
    )

    feature_extractor = FeatureExtractor(
        calibrator=perception_pipeline.calibrator if calibration_data else None,
        velocity_alpha=0.7,
        acceleration_alpha=0.7,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames at {fps:.2f} fps")

    # Process frames
    frame_id = 0
    processed_frames = 0
    all_features: List[np.ndarray] = []
    all_timestamps: List[float] = []
    all_frame_ids: List[int] = []
    all_audio_flags: List[np.ndarray] = []

    # Video writer for visualization output
    video_writer = None
    if enable_visualization:
        cap_for_viz = cv2.VideoCapture(video_path)
        frame_width_viz = int(cap_for_viz.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_viz = int(cap_for_viz.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_output_path = f"{output_path}_visualization.mp4"
        video_writer = cv2.VideoWriter(viz_output_path, fourcc, fps, (frame_width_viz, frame_height_viz))
        logger.info(f"Visualization output: {viz_output_path}")
        cap_for_viz.release()

    logger.info("Processing video...")
    start_time = time.time()

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if requested
            if skip_frames > 0 and processed_frames > 0:
                pbar.update(1)
                frame_id += 1
                for _ in range(skip_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_id += 1
                if not ret:
                    break

            timestamp = frame_id / fps if fps > 0 else 0.0

            # Start health monitoring for this frame
            health_monitor.start_frame(frame_id, timestamp)

            if monitor is not None:
                with monitor.stage("perception"):
                    frame_data = perception_pipeline.process_frame(
                        frame=frame,
                        timestamp=timestamp,
                        frame_id=frame_id,
                    )

                with monitor.stage("recognition"):
                    features = feature_extractor.extract_frame_features(
                        frame=frame_data,
                        frame_width=frame_width,
                    )
            else:
                # Perception: Pose estimation, tracking, audio
                frame_data = perception_pipeline.process_frame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_id=frame_id,
                )

                # Recognition: Feature extraction
                features = feature_extractor.extract_frame_features(
                    frame=frame_data,
                    frame_width=frame_width,
                )

            # Record health metrics
            confidences = [kp.conf for pose in frame_data.poses for kp in pose.keypoints]
            health_issues = health_monitor.end_frame(
                n_detections=len(frame_data.poses),
                confidences=confidences,
            )
            if health_issues:
                for issue in health_issues:
                    logger.debug(f"Frame {frame_id} health issue: {issue}")

            all_features.append(features)
            all_timestamps.append(timestamp)
            all_frame_ids.append(frame_id)
            all_audio_flags.append(
                np.array([1.0 if frame_data.audio_event else 0.0, 0.0])
            )

            if enable_visualization and video_writer is not None:
                # Draw skeleton overlay on frame
                vis_frame = frame.copy()
                vis_frame = draw_frame_overlay(vis_frame, frame_data.poses, min_conf=0.3)
                vis_frame = draw_info_overlay(
                    vis_frame,
                    frame_id=frame_id,
                    fps=fps,
                    n_fencers=len(frame_data.poses),
                )
                video_writer.write(vis_frame)

            processed_frames += 1
            frame_id += 1
            pbar.update(1)

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
        logger.info(f"Visualization saved to {viz_output_path}")

    elapsed_time = time.time() - start_time
    logger.info(
        f"Processed {processed_frames} frames in {elapsed_time:.2f}s "
        f"({processed_frames/elapsed_time:.2f} fps)"
    )

    # Stop profiling and print results
    if monitor is not None:
        monitor.stop()
        logger.info("\n" + monitor.summary())

        # Check if meets target
        if monitor.meets_target(500):  # 500ms for now
            logger.info("✓ Meets 500ms latency target")
        else:
            logger.warning("✗ Exceeds 500ms latency target")

    # Print health summary
    logger.info("\n" + health_monitor.summary())

    # Build feature matrix
    feature_matrix = FeatureMatrix(
        features=np.array(all_features, dtype=np.float32),
        timestamps=all_timestamps,
        frame_ids=all_frame_ids,
        audio_flags=np.array(all_audio_flags, dtype=np.float32),
    )

    # Save output
    metadata = {
        "video_path": video_path,
        "processing_time_seconds": elapsed_time,
        "fps": fps,
        "frame_width": frame_width,
        "confidence_threshold": confidence_threshold,
        "device": device,
        "skip_frames": skip_frames,
    }
    save_features(output_path, feature_matrix, metadata)

    # Export heatmap if requested
    if enable_heatmap:
        logger.info("Exporting feature heatmap...")
        export_feature_heatmap(feature_matrix.features, output_path)
        logger.info(f"Heatmap exported to {output_path}_*.png")

    return feature_matrix


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FencerAI Feature Extraction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input video: {args.video}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Confidence threshold: {args.confidence_threshold}")

    try:
        feature_matrix = process_video(
            video_path=args.video,
            output_path=args.output,
            frame_width=args.frame_width,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            skip_frames=args.skip_frames,
            enable_visualization=args.visualize,
            calibration_file=args.calibration_file,
            enable_profiling=args.profile,
            enable_heatmap=args.heatmap,
        )

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Feature matrix shape: {feature_matrix.features.shape}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
