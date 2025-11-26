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

"""Unit tests for out-of-bound landmark removal functionality."""

from unittest.mock import MagicMock

import pytest

from asl_data_pipeline.models.s3 import InputType, S3Config
from asl_data_pipeline.preprocessing.pose_extractor.extractor import PoseExtractor
from asl_data_pipeline.preprocessing.pose_extractor.models import PoseExtractorConfig, PoseLandmark


class TestOutOfBoundLandmarkRemoval:
    """Test out-of-bound removal for pose landmarks."""

    @pytest.fixture
    def pose_extractor(self, tmp_path, monkeypatch):
        """Create a pose extractor for testing."""
        mock_s3_client = MagicMock()
        monkeypatch.setattr(
            "asl_data_pipeline.preprocessing.pose_extractor.extractor.S3Client",
            lambda config: mock_s3_client,
        )

        config = PoseExtractorConfig(
            s3=S3Config(
                input_uri="s3://dummy-bucket/manifest.jsonl",
                output_uri="s3://dummy-bucket/pose-landmarks/",
                input_type=InputType.MANIFEST,
            ),
            temp_dir=tmp_path,
            model_complexity=0,
            out_of_bound_removal=True,
        )
        return PoseExtractor(config)

    def test_out_of_bound_landmark_removal(self, pose_extractor):
        """Test removal of landmarks outside 1080x1080 image bounds."""
        # Given: mix of valid and out-of-bound landmarks for 1080x1080 image
        landmarks = [
            # VALID landmarks (within 1080x1080 bounds)
            PoseLandmark(x=500, y=400, z=0.1, label="NOSE", visibility=0.9, presence=1.0),
            PoseLandmark(x=1079, y=1079, z=0.2, label="LEFT_SHOULDER", visibility=0.8, presence=1.0),
            PoseLandmark(x=0, y=0, z=0.3, label="RIGHT_SHOULDER", visibility=0.7, presence=1.0),
            # OUT-OF-BOUND landmarks (outside 1080x1080 bounds)
            PoseLandmark(x=1080, y=500, z=0.4, label="LEFT_ELBOW", visibility=0.6, presence=1.0),
            PoseLandmark(x=500, y=1080, z=0.5, label="RIGHT_ELBOW", visibility=0.5, presence=1.0),
            PoseLandmark(x=-10, y=500, z=0.6, label="LEFT_WRIST", visibility=0.4, presence=1.0),
            PoseLandmark(x=500, y=-20, z=0.7, label="RIGHT_WRIST", visibility=0.3, presence=1.0),
            PoseLandmark(x=1500, y=1500, z=0.8, label="LEFT_HIP", visibility=0.2, presence=1.0),
        ]
        image_size = (1080, 1080)

        filtered_landmarks = pose_extractor._remove_out_of_bound_landmarks(landmarks, image_size)

        assert len(filtered_landmarks) == 3

        # Check that the correct landmarks were kept
        kept_labels = {lm.label for lm in filtered_landmarks}
        expected_labels = {"NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER"}
        assert kept_labels == expected_labels
