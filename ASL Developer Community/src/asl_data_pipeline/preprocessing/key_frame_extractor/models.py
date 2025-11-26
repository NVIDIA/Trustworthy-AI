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

"""Pydantic models for key frame extractor."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings

from asl_data_pipeline.models.manifest import ManifestItem
from asl_data_pipeline.models.s3 import S3Config


class ExtractionParameters(BaseModel):
    """Configuration parameters for keyframe extraction."""

    model_config = ConfigDict(frozen=True)

    peak_prominence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum prominence for detecting significant motion peaks",
    )
    active_region_buffer: int = Field(
        default=5,
        ge=0,
        description="Buffer frames to include before and after active region",
    )
    trough_distance: int = Field(
        default=5,
        ge=1,
        description="Minimum distance between motion troughs",
    )
    trough_window_size: int = Field(
        default=3,
        ge=1,
        description="Window size for refining trough positions with sharpness",
    )


class KeyFrameExtractorConfig(BaseSettings):
    """Main configuration for the KeyFrameExtractor."""

    model_config = ConfigDict(
        env_prefix="KEYFRAME_",
        case_sensitive=False,
        frozen=True,
    )

    # S3 Configuration
    s3: S3Config

    # Extraction parameters
    extraction_params: ExtractionParameters = Field(default_factory=ExtractionParameters)

    # Processing configuration
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum number of parallel workers for processing",
    )
    max_videos: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of videos to process (for testing)",
    )
    temp_dir: Optional[Path] = Field(
        default=None,
        description="Temporary directory for processing (uses system temp if None)",
    )
    enable_charts: bool = Field(
        default=True,
        description="Whether to generate motion analysis charts",
    )
    force_write: bool = Field(
        default=False,
        description="Whether to overwrite existing output files",
    )

    # Video file extensions to process
    video_extensions: List[str] = Field(
        default=["webm", "mp4", "mov", "avi"],
        description="Video file extensions to process",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "KeyFrameExtractorConfig":
        """Validate the configuration."""
        if self.extraction_params.trough_window_size % 2 == 0:
            raise ValueError("trough_window_size must be odd")
        return self


class VideoMetadata(BaseModel):
    """Metadata for a processed video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_filename: str
    video_s3_key: str
    total_frame_count: int
    extracted_frame_count: int
    extraction_method: str = "motion_analysis_with_sharpness"
    parameters: ExtractionParameters
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    motion_scores: Optional[List[float]] = Field(
        default=None,
        description="Motion scores for each frame (optional, for debugging)",
    )
    active_region_start: int
    active_region_end: int
    keyframe_indices: List[int]


class KeyFrame(BaseModel):
    """Represents an extracted keyframe."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    is_active_region: bool
    sharpness_score: float
    s3_key: str
    filename: str


class ExtractionResult(BaseModel):
    """Result of keyframe extraction for a single video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_metadata: VideoMetadata
    keyframes: List[KeyFrame]
    manifest_items: List[ManifestItem]
    chart_s3_key: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class BatchExtractionResult(BaseModel):
    """Result of batch keyframe extraction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_videos: int
    successful_extractions: int
    failed_extractions: int
    results: List[ExtractionResult]
    manifest_s3_key: Optional[str] = None
    processing_start_time: datetime
    processing_end_time: Optional[datetime] = None
    total_processing_time_seconds: Optional[float] = None

    @model_validator(mode="after")
    def calculate_timing(self) -> "BatchExtractionResult":
        """Calculate processing time if end time is set."""
        if self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            self.total_processing_time_seconds = delta.total_seconds()
        return self
