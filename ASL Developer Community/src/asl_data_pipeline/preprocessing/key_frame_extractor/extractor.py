# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main KeyFrameExtractor class for production video processing."""

import logging
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.key_frame_extractor.models import (
    BatchExtractionResult,
    ExtractionParameters,
    ExtractionResult,
    KeyFrame,
    KeyFrameExtractorConfig,
    ManifestItem,
    VideoMetadata,
)
from asl_data_pipeline.utils.image_processing import rotate_image
from asl_data_pipeline.utils.s3_client import S3Client, S3ClientError
from asl_data_pipeline.utils.video_utils import get_video_rotation

logger = logging.getLogger(__name__)


class KeyFrameExtractorError(Exception):
    """Custom exception for KeyFrameExtractor errors."""


class KeyFrameExtractor:
    """Production-grade keyframe extractor with S3 integration.

    This class extracts keyframes from videos using motion analysis and sharpness
    measurement. It supports batch processing, S3 storage, and parallel execution.
    """

    def __init__(self, config: KeyFrameExtractorConfig) -> None:
        """Initialize the KeyFrameExtractor.

        Args:
            config: Configuration object for the extractor

        Raises:
            KeyFrameExtractorError: If initialization fails
        """
        if config.s3.input_type != InputType.FOLDER:
            raise KeyFrameExtractorError("KeyFrameExtractor only supports FOLDER input type.")
        self.config = config
        self.s3_client = S3Client(config.s3)

        # Check if buckets exist
        try:
            self.s3_client.check_bucket_exists(self.s3_client.input_bucket)
            self.s3_client.check_bucket_exists(self.s3_client.output_bucket)
        except S3ClientError as e:
            raise KeyFrameExtractorError(f"Failed to initialize S3: {e}") from e

        # Set up temporary directory
        if config.temp_dir:
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = config.temp_dir
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "keyframe_extractor"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "KeyFrameExtractor initialized with input_uri: %s, output_uri: %s, temp_dir: %s, max_workers: %s",
            config.s3.input_uri,
            config.s3.output_uri,
            self._temp_dir,
            config.max_workers,
        )

    def __getstate__(self):
        """Prepare the instance for pickling, excluding the S3 client."""
        state = self.__dict__.copy()
        # The S3 client is not pickleable, so we remove it from the state
        del state["s3_client"]
        return state

    def __setstate__(self, state):
        """Restore the instance after unpickling and re-initialize the S3 client."""
        self.__dict__.update(state)
        # Each process needs its own S3 client instance
        self.s3_client = S3Client(self.config.s3)

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate the sharpness of an image using the variance of the Laplacian.

        Args:
            image: Input image as numpy array

        Returns:
            Sharpness score as float

        Raises:
            KeyFrameExtractorError: If sharpness calculation fails
        """
        if image is None:
            return 0.0

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Compute the Laplacian of the image and then the variance
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(variance)
        except Exception as e:
            raise KeyFrameExtractorError(f"Failed to calculate sharpness: {e}") from e

    def _get_output_dir_key(self, video_s3_key: str) -> str:
        """Get the output directory key for a video."""
        video_name = Path(video_s3_key).stem
        return f"{self.s3_client.output_prefix}{video_name}/"

    def _get_artifacts_dir_key(self, video_s3_key: str) -> str:
        """Get the artifacts directory key for a video."""
        return f"{self._get_output_dir_key(video_s3_key)}artifacts/"

    def extract_keyframes_from_video(
        self,
        video_path: Path,
        params: Optional[ExtractionParameters] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], List[float], int, int]:
        """Extract keyframes from a single video file.

        Args:
            video_path: Path to the video file
            params: Extraction parameters (uses config default if None)

        Returns:
            Tuple containing:
                - Active region keyframes
                - Inactive region keyframes
                - Active region keyframe indices
                - Inactive region keyframe indices
                - Motion scores
                - Active region start
                - Active region end

        Raises:
            KeyFrameExtractorError: If keyframe extraction fails
        """
        params = params or self.config.extraction_params

        try:
            rotation_angle = get_video_rotation(video_path)
            # Open video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise KeyFrameExtractorError(f"Could not open video: {video_path}")

            # Read the first frame
            ret, prev_frame = cap.read()
            if not ret:
                raise KeyFrameExtractorError(f"Could not read first frame: {video_path}")

            if rotation_angle:
                prev_frame = rotate_image(prev_frame, -rotation_angle)

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            motion_scores = []
            frames = [prev_frame]

            logger.debug("Starting motion analysis for %d total frames", frame_count)

            # Calculate motion scores for all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if rotation_angle:
                    frame = rotate_image(frame, -rotation_angle)

                frames.append(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate Dense Optical Flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)

                # Calculate magnitude of the flow vectors
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Store the mean magnitude of motion
                motion_scores.append(np.mean(magnitude))
                prev_gray = gray

            cap.release()

            if not motion_scores:
                raise KeyFrameExtractorError("No motion scores calculated")

            motion_scores = np.array(motion_scores)
            logger.debug("Motion analysis complete. Found %d motion scores.", len(motion_scores))

            # Detect active region
            active_region_start, active_region_end = self._detect_active_region(motion_scores, params)

            # Find stable poses (troughs) within and outside active region
            all_trough_indices, _ = find_peaks(-motion_scores, distance=params.trough_distance)

            # Filter troughs by region
            active_trough_indices = [
                idx for idx in all_trough_indices if active_region_start <= idx <= active_region_end
            ]

            inactive_trough_indices = [
                idx for idx in all_trough_indices if not (active_region_start <= idx <= active_region_end)
            ]

            logger.debug(
                "Trough detection complete. Total troughs: %d, active: %d, inactive: %d",
                len(all_trough_indices),
                len(active_trough_indices),
                len(inactive_trough_indices),
            )

            # Refine keyframes with sharpness measurement
            active_keyframes, active_keyframe_indices = self._refine_keyframes_with_sharpness(
                frames, active_trough_indices, params
            )

            inactive_keyframes, inactive_keyframe_indices = self._refine_keyframes_with_sharpness(
                frames, inactive_trough_indices, params
            )

            logger.debug(
                "Keyframe extraction complete. Active keyframes: %d, inactive keyframes: %d",
                len(active_keyframes),
                len(inactive_keyframes),
            )

            return (
                active_keyframes,
                inactive_keyframes,
                active_keyframe_indices,
                inactive_keyframe_indices,
                motion_scores.tolist(),
                active_region_start,
                active_region_end,
            )

        except Exception as e:
            raise KeyFrameExtractorError(f"Failed to extract keyframes from {video_path}: {e}") from e

    def _detect_active_region(self, motion_scores: np.ndarray, params: ExtractionParameters) -> Tuple[int, int]:
        """Detect the active region in motion scores.

        Args:
            motion_scores: Array of motion scores
            params: Extraction parameters

        Returns:
            Tuple of (start_index, end_index) for active region
        """
        # Find all peaks with minimum prominence
        significant_peak_indices, _ = find_peaks(motion_scores, prominence=params.peak_prominence)

        active_region_start = 0
        active_region_end = len(motion_scores) - 1

        if significant_peak_indices.size > 0:
            # Define the start and end based on first and last significant peaks
            first_peak = significant_peak_indices[0]
            last_peak = significant_peak_indices[-1]

            # Apply buffer to include lead-in and cool-down motions
            active_region_start = max(0, first_peak - params.active_region_buffer)
            active_region_end = min(len(motion_scores) - 1, last_peak + params.active_region_buffer)

            logger.debug(
                "Active region detected. Start: %d, end: %d, peaks: %d",
                active_region_start,
                active_region_end,
                len(significant_peak_indices),
            )
        else:
            logger.debug("No significant peaks found, using entire video as active region")

        return active_region_start, active_region_end

    def _refine_keyframes_with_sharpness(
        self,
        frames: List[np.ndarray],
        trough_indices: List[int],
        params: ExtractionParameters,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Refine keyframe selection using sharpness measurement.

        Args:
            frames: List of all frames
            trough_indices: Indices of detected troughs
            params: Extraction parameters

        Returns:
            Tuple of (selected_frames, selected_indices)
        """
        keyframes = []
        keyframe_indices = []

        for idx in trough_indices:
            start = max(0, idx - params.trough_window_size // 2)
            end = min(len(frames), idx + params.trough_window_size // 2 + 1)

            best_frame_idx = -1
            max_sharpness = -1

            for i in range(start, end):
                sharpness = self.calculate_sharpness(frames[i])
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_frame_idx = i

            if best_frame_idx != -1 and best_frame_idx not in keyframe_indices:
                keyframes.append(frames[best_frame_idx])
                keyframe_indices.append(best_frame_idx)

        return keyframes, keyframe_indices

    def create_motion_analysis_chart(
        self,
        motion_scores: List[float],
        active_region_start: int,
        active_region_end: int,
        active_keyframe_indices: List[int],
        inactive_keyframe_indices: List[int],
        output_path: Path,
    ) -> None:
        """Create and save a motion analysis visualization chart.

        Args:
            motion_scores: List of motion scores
            active_region_start: Start of active region
            active_region_end: End of active region
            active_keyframe_indices: Indices of active keyframes
            inactive_keyframe_indices: Indices of inactive keyframes
            output_path: Path to save the chart
        """
        try:
            plt.figure(figsize=(15, 7))

            motion_array = np.array(motion_scores)

            # Plot the main motion score line
            plt.plot(motion_scores, label="Motion Score")

            # Draw the shaded active region
            plt.axvspan(active_region_start, active_region_end, color="green", alpha=0.15, label="Active Region")

            # Mark the selected keyframes
            if active_keyframe_indices:
                plt.plot(
                    active_keyframe_indices,
                    motion_array[active_keyframe_indices],
                    "x",
                    color="red",
                    markersize=10,
                    mew=2,
                    label="Active Keyframes",
                )

            # Mark the inactive region keyframes
            if inactive_keyframe_indices:
                plt.plot(
                    inactive_keyframe_indices,
                    motion_array[inactive_keyframe_indices],
                    "x",
                    color="orange",
                    markersize=10,
                    mew=2,
                    label="Inactive Keyframes",
                )

            plt.title("Motion Analysis for Keyframe Extraction")
            plt.xlabel("Frame Number")
            plt.ylabel("Average Motion Magnitude")
            plt.legend()
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close()

            logger.debug("Motion analysis chart created at %s", output_path)

        except Exception as e:
            logger.warning("Failed to create motion analysis chart: %s", str(e))
            # Don't raise exception as chart creation is optional

    def process_video(self, video_s3_key: str) -> ExtractionResult:
        """Process a single video and extract keyframes.

        Args:
            video_s3_key: S3 key of the video to process

        Returns:
            ExtractionResult object with processing results

        Raises:
            KeyFrameExtractorError: If video processing fails
        """
        output_dir_key = self._get_output_dir_key(video_s3_key)
        artifacts_dir_key = self._get_artifacts_dir_key(video_s3_key)
        metadata_s3_key = f"{output_dir_key}metadata.json"

        # Check if output already exists when force_write is False
        if not self.config.force_write and self.s3_client.object_exists(
            bucket=self.s3_client.output_bucket, s3_key=metadata_s3_key
        ):
            logger.info("Skipping video - output already exists: %s", video_s3_key)
            return ExtractionResult(
                video_metadata=VideoMetadata(
                    video_filename=Path(video_s3_key).name,
                    video_s3_key=video_s3_key,
                    total_frame_count=0,
                    extracted_frame_count=0,
                    parameters=self.config.extraction_params,
                    active_region_start=0,
                    active_region_end=0,
                    keyframe_indices=[],
                ),
                keyframes=[],
                success=True,
            )

        temp_video_path = None
        temp_dir = None

        try:
            logger.info("Starting video processing: %s", video_s3_key)

            # Download video to temporary location
            temp_video_path = self.s3_client.download_file_to_temp_dir(
                bucket=self.s3_client.input_bucket, s3_key=video_s3_key, temp_dir=self._temp_dir
            )

            # Create temporary directory for outputs
            temp_dir = (
                self._temp_dir / f"processing_{Path(video_s3_key).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract keyframes
            (
                active_keyframes,
                inactive_keyframes,
                active_keyframe_indices,
                inactive_keyframe_indices,
                motion_scores,
                active_region_start,
                active_region_end,
            ) = self.extract_keyframes_from_video(temp_video_path)

            # Create metadata
            metadata = VideoMetadata(
                video_filename=Path(video_s3_key).name,
                video_s3_key=video_s3_key,
                total_frame_count=len(motion_scores) + 1,  # +1 for the first frame
                extracted_frame_count=len(active_keyframes) + len(inactive_keyframes),
                parameters=self.config.extraction_params,
                motion_scores=motion_scores,
                active_region_start=active_region_start,
                active_region_end=active_region_end,
                keyframe_indices=sorted(active_keyframe_indices + inactive_keyframe_indices),
            )

            # Get output directory key
            artifacts_dir_key = self._get_artifacts_dir_key(video_s3_key)

            # Upload all keyframes to frames/ subdirectory, numbered by their video position
            keyframe_objects = []

            # Combine all keyframes with their region info
            all_keyframes_data = []

            # Add active region keyframes
            for frame, frame_idx in zip(active_keyframes, active_keyframe_indices):
                all_keyframes_data.append((frame, frame_idx, True))

            # Add inactive region keyframes
            for frame, frame_idx in zip(inactive_keyframes, inactive_keyframe_indices):
                all_keyframes_data.append((frame, frame_idx, False))

            # Sort by frame index to maintain order
            all_keyframes_data.sort(key=lambda x: x[1])

            # Upload each frame numbered by its video position
            manifest_items = []
            for frame, frame_idx, is_active in all_keyframes_data:
                # Use 6-digit zero-padded frame index as filename
                filename = f"{frame_idx:06d}.jpg"
                s3_key = f"{artifacts_dir_key}{filename}"

                # Encode frame as JPEG
                _, frame_bytes = cv2.imencode(".jpg", frame)

                self.s3_client.upload_data(
                    data=frame_bytes.tobytes(), bucket=self.s3_client.output_bucket, s3_key=s3_key
                )

                keyframe_obj = KeyFrame(
                    frame_index=frame_idx,
                    is_active_region=is_active,
                    sharpness_score=self.calculate_sharpness(frame),
                    s3_key=s3_key,
                    filename=filename,
                )

                keyframe_objects.append(keyframe_obj)
                manifest_items.append(
                    ManifestItem(
                        file_uri=f"s3://{self.s3_client.output_bucket}/{s3_key}",
                        metadata_file=f"s3://{self.s3_client.output_bucket}/{metadata_s3_key}",
                    )
                )

            # Create and upload motion analysis chart
            chart_s3_key = None
            if self.config.enable_charts:
                chart_path = temp_dir / "motion_analysis.png"
                self.create_motion_analysis_chart(
                    motion_scores,
                    active_region_start,
                    active_region_end,
                    active_keyframe_indices,
                    inactive_keyframe_indices,
                    chart_path,
                )

                if chart_path.exists():
                    chart_s3_key = f"{output_dir_key}motion_analysis.png"
                    self.s3_client.upload_file(
                        local_path=chart_path, bucket=self.s3_client.output_bucket, s3_key=chart_s3_key
                    )

            # Upload metadata
            self.s3_client.upload_json_as_dict(
                data=metadata.model_dump(mode="json"),
                bucket=self.s3_client.output_bucket,
                s3_key=metadata_s3_key,
            )

            logger.info(
                "Video processing complete for %s. Active keyframes: %d, inactive keyframes: %d. Output: %s",
                video_s3_key,
                len(active_keyframes),
                len(inactive_keyframes),
                artifacts_dir_key,
            )

            return ExtractionResult(
                video_metadata=metadata,
                keyframes=keyframe_objects,
                manifest_items=manifest_items,
                chart_s3_key=chart_s3_key,
                success=True,
            )

        except Exception as e:
            logger.error("Video processing failed for %s: %s", video_s3_key, str(e))
            return ExtractionResult(
                video_metadata=VideoMetadata(
                    video_filename=Path(video_s3_key).name,
                    video_s3_key=video_s3_key,
                    total_frame_count=0,
                    extracted_frame_count=0,
                    parameters=self.config.extraction_params,
                    active_region_start=0,
                    active_region_end=0,
                    keyframe_indices=[],
                ),
                keyframes=[],
                manifest_items=[],
                success=False,
                error_message=str(e),
            )
        finally:
            # Cleanup
            if temp_video_path and temp_video_path.exists():
                temp_video_path.unlink()
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning("Failed to cleanup temp directory %s: %s", temp_dir, str(e))

    def process_videos_batch(
        self,
        video_s3_keys: Optional[List[str]] = None,
    ) -> BatchExtractionResult:
        """Process multiple videos in parallel.

        Args:
            video_s3_keys: List of S3 keys to process (if None, processes all videos in bucket)

        Returns:
            BatchExtractionResult with processing results
        """
        start_time = datetime.now(timezone.utc)

        # Check if output folder exists when force_write is False
        if not self.config.force_write:
            # List objects with the output prefix to check if folder exists and has content
            existing_objects = self.s3_client.list_files(
                bucket=self.s3_client.output_bucket, prefix=self.s3_client.output_prefix
            )
            if existing_objects:
                raise KeyFrameExtractorError(
                    f"Output folder already exists at {self.config.s3.output_uri} "
                    f"and contains {len(existing_objects)} objects. Set force_write=True to overwrite."
                )

        # Get list of videos to process
        if video_s3_keys is None:
            all_files = self.s3_client.list_files(
                bucket=self.s3_client.input_bucket, prefix=self.s3_client.input_prefix
            )
            video_s3_keys = [
                key for key in all_files if any(key.lower().endswith(f".{ext}") for ext in self.config.video_extensions)
            ]

        # Limit number of videos if max_videos is set
        if self.config.max_videos:
            video_s3_keys = video_s3_keys[: self.config.max_videos]

        if not video_s3_keys:
            logger.warning("No videos found to process")
            return BatchExtractionResult(
                total_videos=0,
                successful_extractions=0,
                failed_extractions=0,
                results=[],
                processing_start_time=start_time,
                processing_end_time=datetime.now(timezone.utc),
            )

        logger.info(
            "Starting batch processing for %d videos with %d max workers",
            len(video_s3_keys),
            self.config.max_workers,
        )

        results = []
        successful_count = 0
        failed_count = 0

        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_key = {executor.submit(self.process_video, video_key): video_key for video_key in video_s3_keys}

            # Collect results as they complete
            for future in as_completed(future_to_key):
                video_key = future_to_key[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        successful_count += 1
                    else:
                        failed_count += 1

                    logger.info(
                        "Video %s processed. Success: %s. Progress: %d/%d",
                        video_key,
                        result.success,
                        len(results),
                        len(video_s3_keys),
                    )

                except Exception as e:
                    logger.error("Unexpected error processing video %s: %s", video_key, str(e))
                    failed_count += 1

        end_time = datetime.now(timezone.utc)

        # Aggregate all manifest items
        all_manifest_items = []
        for result in results:
            if result.success:
                all_manifest_items.extend(result.manifest_items)

        # Upload manifest file
        manifest_s3_key = None
        if all_manifest_items:
            manifest_content = "\n".join([item.model_dump_json() for item in all_manifest_items])
            manifest_s3_key = f"{self.s3_client.output_prefix}manifest.jsonl"
            self.s3_client.upload_data(
                data=manifest_content.encode("utf-8"),
                bucket=self.s3_client.output_bucket,
                s3_key=manifest_s3_key,
            )

        batch_result = BatchExtractionResult(
            total_videos=len(video_s3_keys),
            successful_extractions=successful_count,
            failed_extractions=failed_count,
            results=results,
            manifest_s3_key=manifest_s3_key,
            processing_start_time=start_time,
            processing_end_time=end_time,
        )

        logger.info(
            "Batch processing complete. Total videos: %d, successful: %d, failed: %d, total time: %s",
            batch_result.total_videos,
            batch_result.successful_extractions,
            batch_result.failed_extractions,
            batch_result.total_processing_time_seconds,
        )

        # Upload job metadata
        job_metadata_s3_key = f"{self.s3_client.output_prefix}job_metadata.json"
        job_metadata = batch_result.model_dump(
            mode="json",
            exclude={
                "results": {"__all__": {"video_metadata": {"motion_scores", "keyframe_indices", "manifest_items"}}}
            },
        )
        self.s3_client.upload_json_as_dict(
            data=job_metadata,
            bucket=self.s3_client.output_bucket,
            s3_key=job_metadata_s3_key,
        )

        return batch_result
