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

"""Unit tests for the HandLandmarkExtractor."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2

from asl_data_pipeline.models.s3 import InputType, S3Config
from asl_data_pipeline.preprocessing.hand_landmark_extractor.extractor import HandLandmarkExtractor
from asl_data_pipeline.preprocessing.hand_landmark_extractor.models import HandLandmarkExtractorConfig


def test_hand_landmark_extractor_happy_path(tmp_path, monkeypatch):
    """Test the happy path for the HandLandmarkExtractor."""
    # Given
    mock_s3_client = MagicMock()
    monkeypatch.setattr(
        "asl_data_pipeline.preprocessing.hand_landmark_extractor.extractor.S3Client",
        lambda config: mock_s3_client,
    )

    config = HandLandmarkExtractorConfig(
        s3=S3Config(
            input_uri="s3://dummy-bucket/manifest.jsonl",
            output_uri="s3://dummy-bucket/hand-landmarks/",
            input_type=InputType.MANIFEST,
        ),
        temp_dir=tmp_path,
    )
    extractor = HandLandmarkExtractor(config)

    image_path = Path("tests/data/active_1.jpg")
    image = cv2.imread(str(image_path))

    # When
    status, num_hands, landmarks, _ = extractor._extract_landmarks_from_image(image)

    # Then
    assert status == "success"
    assert num_hands > 0
    assert len(landmarks) > 0
