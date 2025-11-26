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

"""Pydantic models for the FaceLandmarkExtractor."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from asl_data_pipeline.models.s3 import S3Config


class FaceLandmark(BaseModel):
    """Represents a single face landmark."""

    x: float = Field(description="The normalized x-coordinate of the landmark (0.0 to 1.0).")
    y: float = Field(description="The normalized y-coordinate of the landmark (0.0 to 1.0).")
    z: float = Field(description="The z-coordinate of the landmark (depth).")
    landmark_id: int = Field(description="The ID of the landmark (0 to 467 for face mesh).")
    visibility: Optional[float] = Field(None, description="The visibility of the landmark.")
    presence: Optional[float] = Field(None, description="The presence of the landmark.")


class FaceBlendShape(BaseModel):
    """Represents a face blend shape coefficient."""

    category_name: str = Field(description="The name of the blend shape category.")
    score: float = Field(description="The score/weight of the blend shape (0.0 to 1.0).")


class FaceLandmarkMetadata(BaseModel):
    """Metadata for a processed frame's face landmarks."""

    frame_source_s3_key: str = Field(description="The S3 key of the original frame image.")
    landmarks_json_s3_key: str = Field(description="S3 key for the landmarks JSON file.")
    annotated_image_s3_key: Optional[str] = Field(None, description="S3 key for the annotated image.")
    num_faces: int = Field(description="Number of faces detected.")
    has_blendshapes: bool = Field(False, description="Whether blend shapes were extracted.")
    has_transformation_matrix: bool = Field(False, description="Whether transformation matrix was extracted.")


class FaceLandmarkExtractorConfig(BaseModel):
    """Configuration for the FaceLandmarkExtractor."""

    s3: S3Config
    temp_dir: Optional[Path] = Field(None, description="Optional temporary directory for local processing.")
    max_workers: int = Field(10, description="Maximum number of parallel workers.")
    save_annotated_images: bool = Field(True, description="Whether to save annotated images.")
    min_detection_confidence: float = Field(0.5, description="Minimum detection confidence for MediaPipe Face.")
    min_tracking_confidence: float = Field(0.5, description="Minimum tracking confidence for MediaPipe Face.")
    output_face_blendshapes: bool = Field(True, description="Whether to output face blend shapes.")
    output_facial_transformation_matrixes: bool = Field(
        True, description="Whether to output facial transformation matrices."
    )
    num_faces: int = Field(1, description="Maximum number of faces to detect.")
    out_of_bound_removal: bool = Field(True, description="Filter out landmarks that are out of bounds.")
    force_write: bool = Field(False, description="Whether to overwrite existing output files.")


class BatchFaceLandmarkResult(BaseModel):
    """Final result of a batch face landmark processing job."""

    total_frames: int = Field(description="Total number of frames processed.")
    successful_extractions: int = Field(description="Number of frames with successfully detected faces.")
    no_faces_detected: int = Field(description="Number of frames where no faces were detected.")
    unreadable_frames: int = Field(description="Number of frames that could not be read.")
    manifest_s3_key: Optional[str] = Field(None, description="S3 key for the output manifest file.")
    frame_results: List[FaceLandmarkMetadata] = Field(
        default_factory=list, description="List of metadata for each processed frame."
    )
    processing_start_time: datetime = Field(description="Start time of the batch processing job.")
    processing_end_time: datetime = Field(description="End time of the batch processing job.")

    @property
    def total_processing_time_seconds(self) -> float:
        """Calculate the total processing time in seconds."""
        return (self.processing_end_time - self.processing_start_time).total_seconds()
