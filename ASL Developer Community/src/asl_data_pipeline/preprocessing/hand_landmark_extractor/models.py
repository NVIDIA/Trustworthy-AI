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

"""Pydantic models for the HandLandmarkExtractor."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from asl_data_pipeline.models.s3 import S3Config


class HandLandmark(BaseModel):
    """Represents a single hand landmark."""

    x: int = Field(description="The x-coordinate of the landmark.")
    y: int = Field(description="The y-coordinate of the landmark.")
    label: str = Field(description="The label of the landmark (e.g., 'WRIST', 'THUMB_TIP').")
    hand: int = Field(description="The index of the hand this landmark belongs to (0 or 1).")


class HandLandmarkMetadata(BaseModel):
    """Metadata for a processed frame's hand landmarks."""

    frame_source_s3_key: str = Field(description="The S3 key of the original frame image.")
    landmarks_json_s3_key: str = Field(description="S3 key for the landmarks JSON file.")
    annotated_image_s3_key: Optional[str] = Field(None, description="S3 key for the annotated image.")
    num_hands: int = Field(description="Number of hands detected.")


class HandLandmarkExtractorConfig(BaseModel):
    """Configuration for the HandLandmarkExtractor."""

    s3: S3Config
    temp_dir: Optional[Path] = Field(None, description="Optional temporary directory for local processing.")
    max_workers: int = Field(10, description="Maximum number of parallel workers.")
    save_annotated_images: bool = Field(True, description="Whether to save annotated images.")
    min_detection_confidence: float = Field(0.3, description="Minimum detection confidence for MediaPipe Hands.")
    out_of_bound_removal: bool = Field(True, description="Filter out landmarks that are out of bounds.")
    force_write: bool = Field(False, description="Whether to overwrite existing output files.")


class BatchHandLandmarkResult(BaseModel):
    """Final result of a batch processing job."""

    total_frames: int = Field(description="Total number of frames processed.")
    successful_extractions: int = Field(description="Number of frames with successfully detected hands.")
    no_hands_detected: int = Field(description="Number of frames where no hands were detected.")
    unreadable_frames: int = Field(description="Number of frames that could not be read.")
    manifest_s3_key: Optional[str] = Field(None, description="S3 key for the output manifest file.")
    frame_results: List[HandLandmarkMetadata] = Field(
        default_factory=list, description="List of metadata for each processed frame."
    )
    processing_start_time: datetime = Field(description="Start time of the batch processing job.")
    processing_end_time: datetime = Field(description="End time of the batch processing job.")

    @property
    def total_processing_time_seconds(self) -> float:
        """Calculate the total processing time in seconds."""
        return (self.processing_end_time - self.processing_start_time).total_seconds()
