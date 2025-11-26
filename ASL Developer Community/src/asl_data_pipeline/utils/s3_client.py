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

"""S3 client for handling file storage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from asl_data_pipeline.preprocessing.key_frame_extractor.models import S3Config
from asl_data_pipeline.utils.s3_utils import parse_s3_uri

logger = logging.getLogger(__name__)


class S3ClientError(Exception):
    """Custom exception for S3 client errors."""


class S3Client:
    """Client for interacting with S3."""

    def __init__(self, config: S3Config):
        """Initialize the S3 client."""
        self.config = config
        self.input_bucket, self.input_prefix = parse_s3_uri(config.input_uri)
        self.output_bucket, self.output_prefix = parse_s3_uri(config.output_uri)

        try:
            session = boto3.Session(
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region_name=config.region_name,
                profile_name=config.aws_profile,
            )
            self.s3 = session.client("s3")
        except NoCredentialsError as e:
            raise S3ClientError("AWS credentials not found.") from e
        except Exception as e:
            raise S3ClientError(f"Failed to initialize S3 client: {e}") from e

    def check_bucket_exists(self, bucket_name: str):
        """Check if an S3 bucket exists."""
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.debug(f"Bucket '{bucket_name}' exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise S3ClientError(f"Bucket '{bucket_name}' not found.") from e
            else:
                raise S3ClientError(f"Failed to check bucket '{bucket_name}': {e}") from e

    def list_files(self, bucket: str, prefix: str) -> List[str]:
        """List files in an S3 prefix."""
        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            raise S3ClientError(f"Failed to list files in '{bucket}/{prefix}': {e}") from e

    def download_file_to_temp_dir(self, bucket: str, s3_key: str, temp_dir: Path) -> Path:
        """Download a file from S3 to a temporary directory."""
        local_path = temp_dir / Path(s3_key).name
        try:
            self.s3.download_file(bucket, s3_key, str(local_path))
            return local_path
        except ClientError as e:
            raise S3ClientError(f"Failed to download file '{s3_key}': {e}") from e

    def download_file_to_bytes(self, bucket: str, s3_key: str) -> bytes:
        """Download an S3 file to a byte string."""
        try:
            response = self.s3.get_object(Bucket=bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            raise S3ClientError(f"Failed to download file '{s3_key}' to bytes: {e}") from e

    def upload_data(self, data: bytes, bucket: str, s3_key: str):
        """Upload raw bytes to S3."""
        try:
            self.s3.put_object(Bucket=bucket, Key=s3_key, Body=data)
        except ClientError as e:
            raise S3ClientError(f"Failed to upload data to '{s3_key}': {e}") from e

    def upload_json_as_dict(self, data: Dict[str, Any], bucket: str, s3_key: str):
        """Upload a dictionary as a JSON file to S3."""
        try:
            self.s3.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json.dumps(data, indent=2),
                ContentType="application/json",
            )
        except ClientError as e:
            raise S3ClientError(f"Failed to upload JSON to '{s3_key}': {e}") from e

    def upload_file(self, local_path: Path, bucket: str, s3_key: str):
        """Upload a local file to S3."""
        try:
            self.s3.upload_file(str(local_path), bucket, s3_key)
        except ClientError as e:
            raise S3ClientError(f"Failed to upload file '{local_path}' to '{s3_key}': {e}") from e

    def object_exists(self, bucket: str, s3_key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.s3.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise S3ClientError(f"Failed to check for object '{s3_key}': {e}") from e
