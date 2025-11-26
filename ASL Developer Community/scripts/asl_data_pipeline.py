#!/usr/bin/env python3

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

"""
Example script showing how to use the ASL Data Pipeline package.

Run this script with:
    poetry run python -m scripts.asl_data_pipeline
"""

import logging
import sys

from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.hand_landmark_extractor.extractor import (
    HandLandmarkExtractor,
    HandLandmarkExtractorConfig,
)
from asl_data_pipeline.preprocessing.hand_sa_converter.converter import HandSAConverter
from asl_data_pipeline.preprocessing.hand_sa_converter.models import HandSAConverterConfig
from asl_data_pipeline.preprocessing.key_frame_extractor.extractor import KeyFrameExtractor, KeyFrameExtractorConfig
from asl_data_pipeline.preprocessing.key_frame_extractor.models import S3Config
from asl_data_pipeline.preprocessing.pose_extractor.extractor import PoseExtractor, PoseExtractorConfig
from asl_data_pipeline.preprocessing.pose_sa_converter.converter import PoseSAConverter
from asl_data_pipeline.preprocessing.pose_sa_converter.models import PoseSAConverterConfig


def main():
    """Main function demonstrating package usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    # Step 1: Frame Extraction
    key_frame_extractor_config = KeyFrameExtractorConfig(
        s3=S3Config(
            input_uri="s3://bucket-name/input-folder/",
            output_uri="s3://bucket-name/output-folder/video_keyframes/",
            aws_profile="aws-profile-name",
            input_type=InputType.FOLDER,
        ),
        max_videos=4,
        force_write=True,
    )
    frame_extractor = KeyFrameExtractor(key_frame_extractor_config)
    frame_extractor.process_videos_batch()

    # Step 2: Hand Landmark Extraction
    hand_landmark_extractor_config = HandLandmarkExtractorConfig(
        s3=S3Config(
            input_uri="s3://bucket-name/output-folder/video_keyframes/manifest.jsonl",
            output_uri="s3://bucket-name/output-folder/video_hand_landmarks_extracted/",
            aws_profile="aws-profile-name",
            input_type=InputType.MANIFEST,
        ),
        max_workers=4,
        force_write=True,
    )
    hand_landmark_extractor = HandLandmarkExtractor(hand_landmark_extractor_config)
    hand_landmark_extractor.process_batch()

    # Step 3: Pose Extraction
    pose_extractor_config = PoseExtractorConfig(
        s3=S3Config(
            input_uri="s3://bucket-name/output-folder/video_keyframes/manifest.jsonl",
            output_uri="s3://bucket-name/output-folder/video_pose_landmarks_extracted/",
            aws_profile="aws-profile-name",
            input_type=InputType.MANIFEST,
        ),
        out_of_bound_removal=True,
        max_workers=4,
        force_write=True,
    )
    pose_extractor = PoseExtractor(pose_extractor_config)
    pose_extractor.process_batch()

    # Step 4: Hand SuperAnnotate format conversion
    hand_sa_converter_config = HandSAConverterConfig(
        s3=S3Config(
            input_uri="s3://bucket-name/output-folder/video_hand_landmarks_extracted/manifest.jsonl",  # noqa: E501
            output_uri="s3://bucket-name/output-folder/video_hand_landmarks_sa/",
            aws_profile="aws-profile-name",
            input_type=InputType.MANIFEST,
        ),
        force_write=True,
        max_workers=4,
    )
    hand_sa_converter = HandSAConverter(hand_sa_converter_config)
    hand_sa_converter.process_batch()

    # Step 5: Pose SuperAnnotate format conversion
    pose_sa_converter_config = PoseSAConverterConfig(
        s3=S3Config(
            input_uri="s3://bucket-name/output-folder/video_pose_landmarks_extracted/manifest.jsonl",  # noqa: E501
            output_uri="s3://bucket-name/output-folder/video_pose_landmarks_sa/",
            aws_profile="aws-profile-name",
            input_type=InputType.MANIFEST,
        ),
        force_write=True,
        max_workers=4,
    )
    pose_sa_converter = PoseSAConverter(pose_sa_converter_config)
    pose_sa_converter.process_batch()


if __name__ == "__main__":
    main()
