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

"""Pydantic model for S3 configuration."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    """Enum for input types."""

    FOLDER = "folder"
    MANIFEST = "manifest"


class S3Config(BaseModel):
    """S3 configuration for input and output."""

    input_uri: str = Field(description="S3 URI for input data.")
    input_type: InputType = Field(
        default=InputType.FOLDER,
        description="Type of input: 'folder' for a directory of files, 'manifest' for a manifest file.",
    )
    output_uri: str = Field(description="S3 URI for output data (e.g., s3://bucket/keyframes/)")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS Secret Access Key")
    region_name: Optional[str] = Field(default="us-east-1", description="AWS region name")
    aws_profile: Optional[str] = Field(default=None, description="AWS profile name to use")
