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

"""Pydantic models for the Hand SuperAnnotate Format Converter."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from asl_data_pipeline.models.s3 import S3Config


class HandSAConverterConfig(BaseModel):
    """Configuration for the Hand SuperAnnotate Format Converter."""

    s3: S3Config
    temp_dir: Optional[Path] = Field(None, description="Optional temporary directory for local processing.")
    force_write: bool = Field(False, description="Whether to overwrite existing output files.")
    max_workers: int = Field(4, description="Maximum number of parallel workers.")


class HandSAConversionResult(BaseModel):
    """Result of a single hand landmark file conversion."""

    input_file: str = Field(description="Input landmark file S3 key.")
    output_sa_file: Optional[str] = Field(None, description="Output SuperAnnotate format file S3 key.")
    status: str = Field(description="Processing status (success, error, skipped).")
    num_landmarks: int = Field(0, description="Number of landmarks processed.")
    num_hands: int = Field(0, description="Number of hands detected.")
    error_message: Optional[str] = Field(None, description="Error message if processing failed.")
