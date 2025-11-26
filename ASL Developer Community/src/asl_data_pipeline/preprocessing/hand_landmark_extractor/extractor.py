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

"""Main HandLandmarkExtractor class for production processing."""

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from asl_data_pipeline.models.manifest import ManifestItem
from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.hand_landmark_extractor.models import (
    BatchHandLandmarkResult,
    HandLandmark,
    HandLandmarkExtractorConfig,
    HandLandmarkMetadata,
)
from asl_data_pipeline.utils.s3_client import S3Client, S3ClientError
from asl_data_pipeline.utils.s3_utils import parse_s3_uri

logger = logging.getLogger(__name__)


class HandLandmarkExtractorError(Exception):
    """Custom exception for HandLandmarkExtractor errors."""


class HandLandmarkExtractor:
    """Production-grade hand landmark extractor with S3 integration."""

    def __init__(self, config: HandLandmarkExtractorConfig) -> None:
        """Initialize the HandLandmarkExtractor."""
        if config.s3.input_type != InputType.MANIFEST:
            raise HandLandmarkExtractorError("HandLandmarkExtractor only supports MANIFEST input type.")
        self.config = config
        self.s3_client = S3Client(config.s3)

        try:
            self.s3_client.check_bucket_exists(self.s3_client.input_bucket)
            self.s3_client.check_bucket_exists(self.s3_client.output_bucket)
        except S3ClientError as e:
            raise HandLandmarkExtractorError(f"Failed to initialize S3: {e}") from e

        if config.temp_dir:
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = config.temp_dir
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "hand_landmark_extractor"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=config.min_detection_confidence,
        )

        logger.info(
            "HandLandmarkExtractor initialized with input_uri: %s, output_uri: %s, temp_dir: %s, max_workers: %s",
            config.s3.input_uri,
            config.s3.output_uri,
            self._temp_dir,
            config.max_workers,
        )

    def _extract_landmarks_from_image(
        self, image: np.ndarray
    ) -> Tuple[str, int, List[HandLandmark], Optional[np.ndarray]]:
        """Detect and extract hand landmarks from a single image."""
        if image is None:
            return "could_not_read", 0, [], None

        try:
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            landmarks_list = []
            num_hands = 0
            annotated_image = None

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                if self.config.save_annotated_images:
                    annotated_image = image.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )

                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        landmarks_list.append(
                            HandLandmark(
                                x=int(landmark.x * w),
                                y=int(landmark.y * h),
                                label=mp.solutions.hands.HandLandmark(i).name,
                                hand=hand_idx,
                            )
                        )

            if landmarks_list:
                return "success", num_hands, landmarks_list, annotated_image
            else:
                return "no_hands", 0, [], None

        except Exception as e:
            raise HandLandmarkExtractorError(f"Failed to process image: {e}") from e

    def _remove_out_of_bound_landmarks(
        self, landmarks: List[HandLandmark], image_size: Tuple[int, int]
    ) -> List[HandLandmark]:
        """Remove landmarks that are out of bounds."""
        height, width = image_size
        filtered_landmarks = []

        for landmark in landmarks:
            x_in_bounds = 0 <= landmark.x < width
            y_in_bounds = 0 <= landmark.y < height

            if x_in_bounds and y_in_bounds:
                filtered_landmarks.append(landmark)

        return filtered_landmarks

    def _get_output_dir_key(self, video_s3_key: str) -> str:
        """Get the output directory key for a video."""
        video_name = Path(video_s3_key).stem
        return f"{self.s3_client.output_prefix}{video_name}/"

    def _get_landmarks_dir_key(self, s3_key: str) -> str:
        """Get the landmarks directory key for a frame."""
        # This assumes the input s3_key is for a frame from the keyframe extractor
        # e.g., keyframe_output/video1/artifacts/000001.jpg
        # and we want to output to hand_landmark_output/video1/landmarks/000001.json
        parts = s3_key.split("/")
        video_name = parts[-3]
        return f"{self.s3_client.output_prefix}{video_name}/landmarks/"

    def _get_annotated_dir_key(self, s3_key: str) -> str:
        """Get the annotated images directory key for a frame."""
        parts = s3_key.split("/")
        video_name = parts[-3]
        return f"{self.s3_client.output_prefix}{video_name}/annotated/"

    def _get_metadata_dir_key(self, s3_key: str) -> str:
        """Get the metadata directory key for a frame."""
        parts = s3_key.split("/")
        video_name = parts[-3]
        return f"{self.s3_client.output_prefix}{video_name}/metadata/"

    def process_frame(
        self, manifest_item: ManifestItem
    ) -> Tuple[str, Optional[ManifestItem], Optional[HandLandmarkMetadata]]:
        """Process a single frame from the manifest."""
        frame_s3_uri = manifest_item.file_uri
        bucket, s3_key = parse_s3_uri(frame_s3_uri)
        filename = Path(s3_key).name

        temp_image_path = None
        try:
            temp_image_path = self.s3_client.download_file_to_temp_dir(
                bucket=bucket, s3_key=s3_key, temp_dir=self._temp_dir
            )
            image = cv2.imread(str(temp_image_path))

            status, num_hands, landmarks, annotated_image = self._extract_landmarks_from_image(image)

            if landmarks and self.config.out_of_bound_removal:
                h, w = image.shape[:2]
                image_size = (h, w)
                original_landmark_count = len(landmarks)
                landmarks = self._remove_out_of_bound_landmarks(landmarks, image_size)
                filtered_landmark_count = len(landmarks)
                removed_count = original_landmark_count - filtered_landmark_count
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} out-of-bound hand landmarks from {filename} (image size: {h}x{w})"
                    )

            annotated_s3_key = None
            if annotated_image is not None and self.config.save_annotated_images:
                annotated_s3_key = f"{self._get_annotated_dir_key(s3_key)}{filename}"
                _, img_bytes = cv2.imencode(".jpg", annotated_image)
                self.s3_client.upload_data(
                    data=img_bytes.tobytes(),
                    bucket=self.s3_client.output_bucket,
                    s3_key=annotated_s3_key,
                )

            if status == "success":
                landmarks_s3_key = f"{self._get_landmarks_dir_key(s3_key)}{Path(filename).stem}.json"
                self.s3_client.upload_json_as_dict(
                    data=[landmark.model_dump() for landmark in landmarks],
                    bucket=self.s3_client.output_bucket,
                    s3_key=landmarks_s3_key,
                )

                metadata = HandLandmarkMetadata(
                    frame_source_s3_key=s3_key,
                    landmarks_json_s3_key=landmarks_s3_key,
                    annotated_image_s3_key=annotated_s3_key,
                    num_hands=num_hands,
                )
                metadata_s3_key = f"{self._get_metadata_dir_key(s3_key)}{Path(filename).stem}.json"
                self.s3_client.upload_json_as_dict(
                    data=metadata.model_dump(mode="json"),
                    bucket=self.s3_client.output_bucket,
                    s3_key=metadata_s3_key,
                )

                output_manifest_item = ManifestItem(
                    file_uri=f"s3://{self.s3_client.output_bucket}/{landmarks_s3_key}",
                    metadata_file=f"s3://{self.s3_client.output_bucket}/{metadata_s3_key}",
                )
                return status, output_manifest_item, metadata

            return status, None, None

        except Exception as e:
            logger.error("Failed to process frame %s: %s", s3_key, e)
            return "error", None, None
        finally:
            if temp_image_path and temp_image_path.exists():
                temp_image_path.unlink()

    def process_batch(self) -> BatchHandLandmarkResult:
        """Process a batch of videos based on a keyframe extraction job metadata file."""
        start_time = datetime.now(timezone.utc)

        # Check if output folder exists when force_write is False
        if not self.config.force_write:
            # List objects with the output prefix to check if folder exists and has content
            existing_objects = self.s3_client.list_files(
                bucket=self.s3_client.output_bucket, prefix=self.s3_client.output_prefix
            )
            if existing_objects:
                raise HandLandmarkExtractorError(
                    f"Output folder already exists at {self.config.s3.output_uri} "
                    f"and contains {len(existing_objects)} objects. Set force_write=True to overwrite."
                )

        try:
            manifest_bucket, manifest_key = parse_s3_uri(self.config.s3.input_uri)
            manifest_content = self.s3_client.download_file_to_bytes(
                bucket=manifest_bucket, s3_key=manifest_key
            ).decode("utf-8")
            manifest_items = [ManifestItem.model_validate_json(line) for line in manifest_content.strip().split("\n")]
        except (S3ClientError, ValueError) as e:
            raise HandLandmarkExtractorError(
                f"Failed to load or parse manifest from {self.config.s3.input_uri}: {e}"
            ) from e

        successful_extractions = 0
        no_hands_detected = 0
        unreadable_frames = 0
        output_manifest_items = []
        frame_results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_frame = {executor.submit(self.process_frame, item): item.file_uri for item in manifest_items}

            for future in as_completed(future_to_frame):
                try:
                    status, manifest_item, metadata = future.result()
                    if status == "success":
                        successful_extractions += 1
                        if manifest_item:
                            output_manifest_items.append(manifest_item)
                        if metadata:
                            frame_results.append(metadata)
                    elif status == "no_hands":
                        no_hands_detected += 1
                    else:
                        unreadable_frames += 1
                except Exception as e:
                    file_uri = future_to_frame[future]
                    logger.error("Frame processing failed for %s: %s", file_uri, e)
                    unreadable_frames += 1

        end_time = datetime.now(timezone.utc)

        # Upload manifest file
        manifest_s3_key = None
        if output_manifest_items:
            manifest_content = "\n".join([item.model_dump_json() for item in output_manifest_items])
            manifest_s3_key = f"{self.s3_client.output_prefix}manifest.jsonl"
            self.s3_client.upload_data(
                data=manifest_content.encode("utf-8"),
                bucket=self.s3_client.output_bucket,
                s3_key=manifest_s3_key,
            )

        batch_result = BatchHandLandmarkResult(
            total_frames=len(manifest_items),
            successful_extractions=successful_extractions,
            no_hands_detected=no_hands_detected,
            unreadable_frames=unreadable_frames,
            manifest_s3_key=manifest_s3_key,
            frame_results=frame_results,
            processing_start_time=start_time,
            processing_end_time=end_time,
        )

        # Upload job metadata
        output_job_metadata_s3_key = f"{self.s3_client.output_prefix}job_metadata.json"
        self.s3_client.upload_json_as_dict(
            data=batch_result.model_dump(mode="json"),
            bucket=self.s3_client.output_bucket,
            s3_key=output_job_metadata_s3_key,
        )

        logger.info(
            "Batch processing complete. Total frames: %d, successful: %d, no hands: %d, unreadable: %d",
            batch_result.total_frames,
            batch_result.successful_extractions,
            batch_result.no_hands_detected,
            batch_result.unreadable_frames,
        )

        return batch_result
