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

"""Unit tests for the FaceLandmarkExtractor."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from asl_data_pipeline.models.s3 import InputType, S3Config

mediapipe_stub = types.ModuleType("mediapipe")
framework_stub = types.ModuleType("mediapipe.framework")
formats_stub = types.ModuleType("mediapipe.framework.formats")
landmark_pb2_stub = types.ModuleType("mediapipe.framework.formats.landmark_pb2")

sys.modules.setdefault("mediapipe", mediapipe_stub)
sys.modules.setdefault("mediapipe.framework", framework_stub)
sys.modules.setdefault("mediapipe.framework.formats", formats_stub)
sys.modules.setdefault("mediapipe.framework.formats.landmark_pb2", landmark_pb2_stub)

from asl_data_pipeline.preprocessing.face_landmark_extractor.models import FaceLandmarkExtractorConfig  # noqa: E402
from asl_data_pipeline.preprocessing.face_landmark_extractor.extractor import (  # noqa: E402
    FaceLandmarkExtractor,
    FaceLandmarkExtractorError,
)


def test_face_landmark_extractor_process_batch_raises_if_output_exists():
    """Test that batch processing refuses to overwrite existing output when force_write is disabled."""
    config = FaceLandmarkExtractorConfig(
        s3=S3Config(
            input_uri="s3://dummy-bucket/manifest.jsonl",
            output_uri="s3://dummy-bucket/face-landmarks/",
            input_type=InputType.MANIFEST,
        ),
        force_write=False,
    )

    extractor = FaceLandmarkExtractor.__new__(FaceLandmarkExtractor)
    extractor.config = config
    extractor.s3_client = MagicMock()
    extractor.s3_client.output_bucket = "dummy-bucket"
    extractor.s3_client.output_prefix = "face-landmarks/"
    extractor.s3_client.list_files.return_value = ["face-landmarks/existing.json"]

    with pytest.raises(FaceLandmarkExtractorError, match="Output folder already exists"):
        extractor.process_batch()

    extractor.s3_client.list_files.assert_called_once_with(
        bucket="dummy-bucket",
        prefix="face-landmarks/",
    )
    extractor.s3_client.download_file_to_bytes.assert_not_called()
