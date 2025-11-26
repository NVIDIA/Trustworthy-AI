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

"""Unit tests for the Hand SuperAnnotate Format Converter."""

from unittest.mock import MagicMock

from asl_data_pipeline.models.s3 import InputType, S3Config
from asl_data_pipeline.preprocessing.hand_sa_converter.converter import HandSAConverter
from asl_data_pipeline.preprocessing.hand_sa_converter.models import HandSAConverterConfig


def test_hand_sa_converter_happy_path(tmp_path, monkeypatch):
    """Test the happy path for the Hand SuperAnnotate Format Converter."""
    # Given
    mock_s3_client = MagicMock()
    monkeypatch.setattr(
        "asl_data_pipeline.preprocessing.hand_sa_converter.converter.S3Client",
        lambda config: mock_s3_client,
    )

    config = HandSAConverterConfig(
        s3=S3Config(
            input_uri="s3://dummy-bucket/manifest.jsonl",
            output_uri="s3://dummy-bucket/sa-output/",
            input_type=InputType.MANIFEST,
        ),
        temp_dir=tmp_path,
    )
    converter = HandSAConverter(config)

    # Sample hand landmark data
    landmarks_data = [
        {"x": 100, "y": 200, "label": "WRIST", "hand": 0},
        {"x": 150, "y": 250, "label": "THUMB_TIP", "hand": 0},
        {"x": 300, "y": 400, "label": "INDEX_TIP", "hand": 1},
    ]

    # When
    image_url = "https://test-bucket.s3.us-east-1.amazonaws.com/test-image.jpg"
    result = converter._convert_landmarks_to_sa_format(landmarks_data, image_url)

    # Then
    assert result["image_url"] == image_url
    assert result["table"]["rows_count"] == 3
    assert len(result["table"]["metadata"]["groupList"]) == 2  # Two hands
    assert len(result["table"]["rows"]) == 3

    # Check group names
    group_names = [group["name"] for group in result["table"]["metadata"]["groupList"]]
    assert "Hand 1" in group_names
    assert "Hand 2" in group_names

    # Check that all landmarks have proper structure
    for row in result["table"]["rows"]:
        assert "step" in row
        assert "vertex" in row
        assert "visibility" in row
        assert "__sa-group__" in row
        assert "points" in row
        assert "x" in row["points"]
        assert "y" in row["points"]
