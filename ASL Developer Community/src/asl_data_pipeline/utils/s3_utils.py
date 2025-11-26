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

"""S3 utility functions."""

from typing import Tuple


class S3UriError(ValueError):
    """Custom exception for S3 URI errors."""


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse an S3 URI into bucket and key.

    Args:
        s3_uri: The S3 URI (e.g., "s3://bucket-name/path/to/key").

    Returns:
        A tuple of (bucket, key).

    Raises:
        S3UriError: If the URI is invalid.
    """
    if not s3_uri.startswith("s3://"):
        raise S3UriError("S3 URI must start with 's3://'")

    parts = s3_uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[0]:
        raise S3UriError("Invalid S3 URI format. Expected 's3://bucket/key'")

    bucket = parts[0]
    key = parts[1]
    return bucket, key
