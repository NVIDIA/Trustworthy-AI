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

"""SuperAnnotate Format Converter for pose landmarks."""

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from asl_data_pipeline.models.manifest import ManifestItem
from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.pose_sa_converter.models import PoseSAConverterConfig
from asl_data_pipeline.utils.s3_client import S3Client, S3ClientError
from asl_data_pipeline.utils.s3_utils import parse_s3_uri

TAXONOMY_URL = "https://superannotatedemo.s3.us-east-1.amazonaws.com/asl-data-pipeline/taxonomy_ref/pose-landmarks.png"

# Mapping from MediaPipe pose landmark labels to SuperAnnotate classList format
MEDIAPIPE_TO_SA_MAPPING = {
    "NOSE": {"id": 1, "name": "nose"},
    "LEFT_EYE_INNER": {"id": 2, "name": "left eye (inner)"},
    "LEFT_EYE": {"id": 3, "name": "left eye"},
    "LEFT_EYE_OUTER": {"id": 4, "name": "left eye (outer)"},
    "RIGHT_EYE_INNER": {"id": 5, "name": "right eye (inner)"},
    "RIGHT_EYE": {"id": 6, "name": "right eye"},
    "RIGHT_EYE_OUTER": {"id": 7, "name": "right eye (outer)"},
    "LEFT_EAR": {"id": 8, "name": "left ear"},
    "RIGHT_EAR": {"id": 9, "name": "right ear"},
    "MOUTH_LEFT": {"id": 10, "name": "mouth (left)"},
    "MOUTH_RIGHT": {"id": 11, "name": "mouth (right)"},
    "LEFT_SHOULDER": {"id": 12, "name": "left shoulder"},
    "RIGHT_SHOULDER": {"id": 13, "name": "right shoulder"},
    "LEFT_ELBOW": {"id": 14, "name": "left elbow"},
    "RIGHT_ELBOW": {"id": 15, "name": "right elbow"},
    "LEFT_WRIST": {"id": 16, "name": "left wrist"},
    "RIGHT_WRIST": {"id": 17, "name": "right wrist"},
    "LEFT_PINKY": {"id": 18, "name": "left pinky"},
    "RIGHT_PINKY": {"id": 19, "name": "right pinky"},
    "LEFT_INDEX": {"id": 20, "name": "left index"},
    "RIGHT_INDEX": {"id": 21, "name": "right index"},
    "LEFT_THUMB": {"id": 22, "name": "left thumb"},
    "RIGHT_THUMB": {"id": 23, "name": "right thumb"},
    "LEFT_HIP": {"id": 24, "name": "left hip"},
    "RIGHT_HIP": {"id": 25, "name": "right hip"},
    "LEFT_KNEE": {"id": 26, "name": "left knee"},
    "RIGHT_KNEE": {"id": 27, "name": "right knee"},
    "LEFT_ANKLE": {"id": 28, "name": "left ankle"},
    "RIGHT_ANKLE": {"id": 29, "name": "right ankle"},
    "LEFT_HEEL": {"id": 30, "name": "left heel"},
    "RIGHT_HEEL": {"id": 31, "name": "right heel"},
    "LEFT_FOOT_INDEX": {"id": 32, "name": "left foot index"},
    "RIGHT_FOOT_INDEX": {"id": 33, "name": "right foot index"},
}

logger = logging.getLogger(__name__)


class PoseSAConverterError(Exception):
    """Custom exception for Pose SuperAnnotate Converter errors."""


class PoseSAConverter:
    """Convert pose landmark JSON to SuperAnnotate format."""

    def __init__(self, config: PoseSAConverterConfig) -> None:
        """Initialize the Pose SuperAnnotate Converter."""
        if config.s3.input_type != InputType.MANIFEST:
            raise PoseSAConverterError("PoseSAConverter only supports MANIFEST input type.")
        self.config = config
        self.s3_client = S3Client(config.s3)

        try:
            self.s3_client.check_bucket_exists(self.s3_client.input_bucket)
            self.s3_client.check_bucket_exists(self.s3_client.output_bucket)
        except S3ClientError as e:
            raise PoseSAConverterError(f"Failed to initialize S3: {e}") from e

        if config.temp_dir:
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = config.temp_dir
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "pose_sa_converter"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PoseSAConverter initialized with input_uri: %s, output_uri: %s, temp_dir: %s, max_workers: %s",
            config.s3.input_uri,
            config.s3.output_uri,
            self._temp_dir,
            config.max_workers,
        )

    def _convert_landmarks_to_sa_format(self, landmarks_data: List[Dict], image_url: str = "") -> Dict:
        """Convert raw pose landmark data to SuperAnnotate format."""
        # All landmarks in a single group (Pose)

        # Get taxonomy URL for pose landmarks
        taxonomy_url = TAXONOMY_URL

        # Create SuperAnnotate format structure
        sa_format = {
            "image_url": image_url,
            "taxonomy": taxonomy_url,
            "table": {"rows": [], "metadata": {"groupList": []}, "rows_count": 0},
        }

        # Single group for all pose landmarks
        group_id = 1
        group_name = "Pose"

        # Add group metadata
        sa_format["table"]["metadata"]["groupList"].append({"id": group_id, "name": group_name})

        # Process all landmarks in a single group
        for landmark in landmarks_data:
            mediapipe_label = landmark.get("label", "")

            # Get the mapped name from the classList
            if mediapipe_label in MEDIAPIPE_TO_SA_MAPPING:
                mapped_data = MEDIAPIPE_TO_SA_MAPPING[mediapipe_label]
                step_name = mapped_data["name"]
                vertex_id = mapped_data["id"]
            else:
                # Fallback for unmapped labels
                step_name = mediapipe_label.lower().replace("_", " ")
                vertex_id = 0
                logger.warning(f"Unmapped pose landmark label: {mediapipe_label}")

            row = {
                "step": step_name,
                "vertex": vertex_id,
                "id": f"group-{group_id}-vertex-{vertex_id}",
                "visibility": True,
                "__sa-group__": group_id,
                "points": {"x": int(landmark["x"]), "y": int(landmark["y"])},
                "movemark": False,
            }
            sa_format["table"]["rows"].append(row)

        sa_format["table"]["rows_count"] = len(sa_format["table"]["rows"])
        return sa_format

    def _get_output_sa_key(self, landmarks_s3_key: str) -> str:
        """Get the output S3 key for the SuperAnnotate format file."""
        # Extract video name and frame name from landmarks key
        parts = landmarks_s3_key.split("/")
        video_name = parts[-3] if len(parts) >= 3 else "unknown"
        frame_name = Path(landmarks_s3_key).stem

        # Create combined filename: video_name + "_" + frame_name
        combined_filename = f"{video_name}_{frame_name}.json"
        return f"{self.s3_client.output_prefix}pose_superannotate/{combined_filename}"

    def process_landmark_file(self, manifest_item: ManifestItem) -> Tuple[str, Optional[str]]:
        """Process a single pose landmark file and convert to SuperAnnotate format."""
        landmarks_s3_uri = manifest_item.file_uri
        metadata_s3_uri = manifest_item.metadata_file

        landmarks_bucket, landmarks_s3_key = parse_s3_uri(landmarks_s3_uri)
        metadata_bucket, metadata_s3_key = parse_s3_uri(metadata_s3_uri)

        filename = Path(landmarks_s3_key).stem

        try:
            # Download landmark data
            landmarks_content = self.s3_client.download_file_to_bytes(
                bucket=landmarks_bucket, s3_key=landmarks_s3_key
            ).decode("utf-8")
            landmarks_data = json.loads(landmarks_content)

            # Download metadata
            metadata_content = self.s3_client.download_file_to_bytes(
                bucket=metadata_bucket, s3_key=metadata_s3_key
            ).decode("utf-8")
            metadata = json.loads(metadata_content)

            # Get image URL from metadata (no copying, just reference)
            frame_source_s3_key = metadata.get("frame_source_s3_key", "")
            image_url = ""

            if frame_source_s3_key:
                # Create HTTPS URL pointing to the original image location
                image_url = f"https://{landmarks_bucket}.s3.us-east-1.amazonaws.com/{frame_source_s3_key}"

            # Convert to SuperAnnotate format with the source image URL
            sa_data = self._convert_landmarks_to_sa_format(landmarks_data, image_url)

            # Copy the converted SuperAnnotate format JSON to output directory
            sa_output_key = self._get_output_sa_key(landmarks_s3_key)
            self.s3_client.upload_json_as_dict(
                data=sa_data,
                bucket=self.s3_client.output_bucket,
                s3_key=sa_output_key,
            )

            logger.info(f"Converted {filename} to SuperAnnotate format")
            return "success", sa_output_key

        except Exception as e:
            logger.error(f"Failed to process landmark file {filename}: {e}")
            return "error", None

    def process_batch(self) -> Dict:
        """Process a batch of hand landmark files based on manifest."""
        start_time = datetime.now(timezone.utc)
        # Check if output folder exists when force_write is False
        if not self.config.force_write:
            existing_objects = self.s3_client.list_files(
                bucket=self.s3_client.output_bucket, prefix=self.s3_client.output_prefix
            )
            if existing_objects:
                raise PoseSAConverterError(
                    f"Output folder already exists at {self.config.s3.output_uri} "
                    f"and contains {len(existing_objects)} objects. Set force_write=True to overwrite."
                )

        try:
            # Load manifest file
            manifest_bucket, manifest_key = parse_s3_uri(self.config.s3.input_uri)
            manifest_content = self.s3_client.download_file_to_bytes(
                bucket=manifest_bucket, s3_key=manifest_key
            ).decode("utf-8")
            manifest_items = [ManifestItem.model_validate_json(line) for line in manifest_content.strip().split("\n")]
        except (S3ClientError, ValueError) as e:
            raise PoseSAConverterError(f"Failed to load or parse manifest from {self.config.s3.input_uri}: {e}") from e

        successful_conversions = 0
        failed_conversions = 0
        output_files = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_item = {
                executor.submit(self.process_landmark_file, item): item.file_uri for item in manifest_items
            }

            for future in as_completed(future_to_item):
                try:
                    status, output_s3_key = future.result()
                    if status == "success":
                        successful_conversions += 1
                        if output_s3_key:
                            output_files.append(output_s3_key)
                    else:
                        failed_conversions += 1
                except Exception as e:
                    file_uri = future_to_item[future]
                    logger.error(f"Landmark file processing failed for {file_uri}: {e}")
                    failed_conversions += 1

        end_time = datetime.now(timezone.utc)

        # Create result summary
        result = {
            "total_files": len(manifest_items),
            "successful_conversions": successful_conversions,
            "failed_conversions": failed_conversions,
            "output_files": output_files,
            "processing_start_time": start_time.isoformat(),
            "processing_end_time": end_time.isoformat(),
            "total_processing_time_seconds": (end_time - start_time).total_seconds(),
        }

        # Upload job metadata
        output_job_metadata_s3_key = f"{self.s3_client.output_prefix}hand_conversion_job_metadata.json"
        self.s3_client.upload_json_as_dict(
            data=result,
            bucket=self.s3_client.output_bucket,
            s3_key=output_job_metadata_s3_key,
        )

        logger.info(
            "Hand SuperAnnotate format conversion complete. Total files: %d, successful: %d, "
            "failed: %d, processing time: %.2f seconds",
            result["total_files"],
            result["successful_conversions"],
            result["failed_conversions"],
            result["total_processing_time_seconds"],
        )

        return result
