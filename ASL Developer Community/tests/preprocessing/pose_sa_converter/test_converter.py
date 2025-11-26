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

"""Unit tests for the Pose SuperAnnotate Format Converter."""

from unittest.mock import MagicMock

from asl_data_pipeline.models.s3 import InputType, S3Config
from asl_data_pipeline.preprocessing.pose_sa_converter.converter import PoseSAConverter
from asl_data_pipeline.preprocessing.pose_sa_converter.models import PoseSAConverterConfig


def test_pose_sa_converter_happy_path(tmp_path, monkeypatch):
    """Test the happy path for the Pose SuperAnnotate Format Converter."""
    mock_s3_client = MagicMock()
    monkeypatch.setattr(
        "asl_data_pipeline.preprocessing.pose_sa_converter.converter.S3Client",
        lambda config: mock_s3_client,
    )

    config = PoseSAConverterConfig(
        s3=S3Config(
            input_uri="s3://dummy-bucket/manifest.jsonl",
            output_uri="s3://dummy-bucket/sa-output/",
            input_type=InputType.MANIFEST,
        ),
        temp_dir=tmp_path,
    )
    converter = PoseSAConverter(config)

    # Sample pose landmark data with face, mouth, and body landmarks
    landmarks_data = [
        {"x": 569, "y": 660, "z": -0.92, "label": "NOSE", "visibility": 0.999},
        {"x": 500, "y": 650, "z": -0.8, "label": "LEFT_EYE", "visibility": 0.95},
        {"x": 400, "y": 700, "z": -0.7, "label": "MOUTH_LEFT", "visibility": 0.9},
        {"x": 450, "y": 705, "z": -0.6, "label": "MOUTH_RIGHT", "visibility": 0.85},
        {"x": 300, "y": 400, "z": 0.1, "label": "LEFT_SHOULDER", "visibility": 0.8},
        {"x": 350, "y": 450, "z": 0.2, "label": "RIGHT_SHOULDER", "visibility": 0.75},
    ]

    image_url = "https://test-bucket.s3.us-east-1.amazonaws.com/test-image.jpg"
    result = converter._convert_landmarks_to_sa_format(landmarks_data, image_url)

    assert result["image_url"] == image_url
    assert result["table"]["rows_count"] == 6
    assert len(result["table"]["metadata"]["groupList"]) == 1
    assert len(result["table"]["rows"]) == 6

    # Check group name - all landmarks in single "Pose" group
    group_names = [group["name"] for group in result["table"]["metadata"]["groupList"]]
    assert "Pose" in group_names
    assert result["table"]["metadata"]["groupList"][0]["id"] == 1

    # Check that all landmarks are in the same group
    pose_rows = [row for row in result["table"]["rows"] if row["__sa-group__"] == 1]
    assert len(pose_rows) == 6

    # Check that all landmarks have proper structure
    for row in result["table"]["rows"]:
        assert "step" in row
        assert "vertex" in row
        assert "visibility" in row
        assert "__sa-group__" in row
        assert row["__sa-group__"] == 1  # All landmarks should be in group 1
        assert "points" in row
        assert "x" in row["points"]
        assert "y" in row["points"]

    # Verify specific landmarks match the classList format
    nose_row = [row for row in result["table"]["rows"] if row["step"] == "nose"][0]
    assert nose_row["vertex"] == 1

    left_eye_row = [row for row in result["table"]["rows"] if row["step"] == "left eye"][0]
    assert left_eye_row["vertex"] == 3

    left_shoulder_row = [row for row in result["table"]["rows"] if row["step"] == "left shoulder"][0]
    assert left_shoulder_row["vertex"] == 12
