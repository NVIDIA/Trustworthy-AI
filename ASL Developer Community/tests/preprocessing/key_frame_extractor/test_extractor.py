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

"""Unit tests for the KeyFrameExtractor."""

from pathlib import Path
from unittest.mock import MagicMock

from asl_data_pipeline.preprocessing.key_frame_extractor.extractor import KeyFrameExtractor
from asl_data_pipeline.preprocessing.key_frame_extractor.models import KeyFrameExtractorConfig, S3Config


def test_key_frame_extractor_happy_path(tmp_path, monkeypatch):
    """Test the happy path for the KeyFrameExtractor."""
    # Given
    mock_s3_client = MagicMock()
    monkeypatch.setattr(
        "asl_data_pipeline.preprocessing.key_frame_extractor.extractor.S3Client",
        lambda config: mock_s3_client,
    )

    config = KeyFrameExtractorConfig(
        s3=S3Config(
            input_uri="s3://dummy-bucket/videos/",
            output_uri="s3://dummy-bucket/keyframes/",
        ),
        temp_dir=tmp_path,
    )
    extractor = KeyFrameExtractor(config)

    video_path = Path("tests/data/5xwRgAwJrmW3VhrBTNhicgWpXS33_1740260341-707000000.webm")

    # When
    (
        active_keyframes,
        inactive_keyframes,
        _,
        _,
        _,
        _,
        _,
    ) = extractor.extract_keyframes_from_video(video_path)

    # Then
    assert len(active_keyframes) > 0
    assert len(inactive_keyframes) >= 0  # There might be no inactive keyframes
