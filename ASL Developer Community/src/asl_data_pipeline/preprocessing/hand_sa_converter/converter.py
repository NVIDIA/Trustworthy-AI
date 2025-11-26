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

"""SuperAnnotate Format Converter for hand landmarks."""

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from asl_data_pipeline.models.manifest import ManifestItem
from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.hand_sa_converter.models import HandSAConverterConfig
from asl_data_pipeline.utils.s3_client import S3Client, S3ClientError
from asl_data_pipeline.utils.s3_utils import parse_s3_uri

# Default taxonomy URLs for different landmark types
TAXONOMY_URL = "https://superannotatedemo.s3.us-east-1.amazonaws.com/asl-data-pipeline/taxonomy_ref/hand-landmarks.png"

logger = logging.getLogger(__name__)


class HandSAConverterError(Exception):
    """Custom exception for Hand SuperAnnotate Converter errors."""


class HandSAConverter:
    """Convert hand landmark JSON to SuperAnnotate format."""

    def __init__(self, config: HandSAConverterConfig) -> None:
        """Initialize the Hand SuperAnnotate Converter."""
        if config.s3.input_type != InputType.MANIFEST:
            raise HandSAConverterError("HandSAConverter only supports MANIFEST input type.")
        self.config = config
        self.s3_client = S3Client(config.s3)

        try:
            self.s3_client.check_bucket_exists(self.s3_client.input_bucket)
            self.s3_client.check_bucket_exists(self.s3_client.output_bucket)
        except S3ClientError as e:
            raise HandSAConverterError(f"Failed to initialize S3: {e}") from e

        if config.temp_dir:
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = config.temp_dir
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "sa_format_converter"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "HandSAConverter initialized with input_uri: %s, output_uri: %s, temp_dir: %s, max_workers: %s",
            config.s3.input_uri,
            config.s3.output_uri,
            self._temp_dir,
            config.max_workers,
        )

    def _convert_landmarks_to_sa_format(self, landmarks_data: List[Dict], image_url: str = "") -> Dict:
        """Convert raw landmark data to SuperAnnotate format."""
        # Group hand landmarks by hand ID
        groups = {}
        for landmark in landmarks_data:
            hand_id = landmark.get("hand", 0)
            if hand_id not in groups:
                groups[hand_id] = []
            groups[hand_id].append(landmark)

        # Get taxonomy URL for hand landmarks
        taxonomy_url = TAXONOMY_URL

        # Create SuperAnnotate format structure
        sa_format = {
            "image_url": image_url,
            "taxonomy": taxonomy_url,
            "table": {"rows": [], "metadata": {"groupList": []}, "rows_count": 0},
        }

        # Process each hand as a separate SuperAnnotate group
        for hand_key in sorted(groups.keys()):
            group_id = hand_key + 1
            group_name = f"Hand {group_id}"

            # Add group metadata
            sa_format["table"]["metadata"]["groupList"].append({"id": group_id, "name": group_name})

            # Add landmarks for this hand with sequential vertex numbers
            for vertex_idx, landmark in enumerate(groups[hand_key], 1):
                row = {
                    "step": landmark.get("label", f"landmark_{vertex_idx}"),
                    "vertex": vertex_idx,
                    "id": f"group-{group_id}-vertex-{vertex_idx}",
                    "visibility": True,
                    "__sa-group__": group_id,
                    "points": {
                        "x": int(landmark["x"]),
                        "y": int(landmark["y"]),
                    },
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
        return f"{self.s3_client.output_prefix}hand_superannotate/{combined_filename}"

    def process_landmark_file(self, manifest_item: ManifestItem) -> Tuple[str, Optional[str]]:
        """Process a single landmark file and convert to SuperAnnotate format."""
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

            # Get image URL from metadata
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
        """Process a batch of landmark files based on manifest."""
        start_time = datetime.now(timezone.utc)

        # Check if output folder exists when force_write is False
        if not self.config.force_write:
            existing_objects = self.s3_client.list_files(
                bucket=self.s3_client.output_bucket, prefix=self.s3_client.output_prefix
            )
            if existing_objects:
                raise HandSAConverterError(
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
            raise HandSAConverterError(f"Failed to load or parse manifest from {self.config.s3.input_uri}: {e}") from e

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
        output_job_metadata_s3_key = f"{self.s3_client.output_prefix}conversion_job_metadata.json"
        self.s3_client.upload_json_as_dict(
            data=result,
            bucket=self.s3_client.output_bucket,
            s3_key=output_job_metadata_s3_key,
        )

        logger.info(
            "SuperAnnotate format conversion complete. Total files: %d, successful: %d, "
            "failed: %d, processing time: %.2f seconds",
            result["total_files"],
            result["successful_conversions"],
            result["failed_conversions"],
            result["total_processing_time_seconds"],
        )

        return result
