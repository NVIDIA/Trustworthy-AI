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

import json
import subprocess
from pathlib import Path
from typing import Optional


def get_video_rotation(video_path: Path) -> Optional[int]:
    """Get the rotation angle of a video from its metadata.

    Args:
        video_path: The path to the video file.

    Returns:
        The rotation angle in degrees (e.g., -90, 90, 180), or None if no rotation is specified.
    """
    try:
        command = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)

        for stream in metadata.get("streams", []):
            if stream.get("codec_type") == "video":
                side_data = stream.get("side_data_list", [])
                for data in side_data:
                    if data.get("side_data_type") == "Display Matrix":
                        return data.get("rotation")
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None

    return None
