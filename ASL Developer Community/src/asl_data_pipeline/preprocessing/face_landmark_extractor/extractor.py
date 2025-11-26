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

"""Main FaceLandmarkExtractor class for production processing."""

import logging
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from asl_data_pipeline.models.manifest import ManifestItem
from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.face_landmark_extractor.models import (
    BatchFaceLandmarkResult,
    FaceBlendShape,
    FaceLandmark,
    FaceLandmarkExtractorConfig,
    FaceLandmarkMetadata,
)
from asl_data_pipeline.utils.s3_client import S3Client, S3ClientError
from asl_data_pipeline.utils.s3_utils import parse_s3_uri

logger = logging.getLogger(__name__)


class FaceLandmarkExtractorError(Exception):
    """Custom exception for FaceLandmarkExtractor errors."""


class FaceLandmarkExtractor:
    """Production-grade face landmark extractor with S3 integration."""

    # MediaPipe face landmarker model URL
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(self, config: FaceLandmarkExtractorConfig) -> None:
        """Initialize the FaceLandmarkExtractor."""
        if config.s3.input_type != InputType.MANIFEST:
            raise FaceLandmarkExtractorError("FaceLandmarkExtractor only supports MANIFEST input type.")
        self.config = config
        self.s3_client = S3Client(config.s3)

        try:
            self.s3_client.check_bucket_exists(self.s3_client.input_bucket)
            self.s3_client.check_bucket_exists(self.s3_client.output_bucket)
        except S3ClientError as e:
            raise FaceLandmarkExtractorError(f"Failed to initialize S3: {e}") from e

        if config.temp_dir:
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = config.temp_dir
        else:
            self._temp_dir = Path(tempfile.gettempdir()) / "face_landmark_extractor"

        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Download face landmarker model if not exists
        model_path = self._download_model()

        # Initialize MediaPipe Face Landmarker
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=config.output_face_blendshapes,
            output_facial_transformation_matrixes=config.output_facial_transformation_matrixes,
            num_faces=config.num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_tracking_confidence,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_landmarker_options)

        logger.info("FaceLandmarkExtractor initialized successfully")

    def _download_model(self) -> Path:
        """Download the face landmarker model if not already present."""
        model_dir = Path.home() / ".mediapipe" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker.task"

        if not model_path.exists():
            logger.info(f"Downloading face landmarker model from {self.MODEL_URL}")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                logger.info(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                raise FaceLandmarkExtractorError(f"Failed to download model: {e}") from e
        else:
            logger.info(f"Using existing model from {model_path}")

        return model_path

    def _get_landmarks_dir_key(self, s3_key: str) -> str:
        """Get the landmarks directory key for a frame."""
        parts = s3_key.split("/")
        video_name = parts[-2]
        return f"{self.s3_client.output_prefix}{video_name}/landmarks/"

    def _get_annotated_dir_key(self, s3_key: str) -> str:
        """Get the annotated images directory key for a frame."""
        parts = s3_key.split("/")
        video_name = parts[-2]
        return f"{self.s3_client.output_prefix}{video_name}/annotated/"

    def _get_metadata_dir_key(self, s3_key: str) -> str:
        """Get the metadata directory key for a frame."""
        parts = s3_key.split("/")
        video_name = parts[-2]
        return f"{self.s3_client.output_prefix}{video_name}/metadata/"

    def _draw_landmarks_on_image(self, rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Draw face landmarks on the image using MediaPipe's drawing utilities."""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize (following official MediaPipe example)
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Convert landmarks to protobuf format for drawing_utils
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in face_landmarks
                ]
            )

            # Draw the face mesh tesselation (the full mesh network)
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Draw face contours (stronger lines for face outline)
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )

            # Draw irises
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image

    def _filter_out_of_bound_landmarks(
        self, landmarks: List[FaceLandmark], image_width: int, image_height: int
    ) -> List[FaceLandmark]:
        """Filter out landmarks that are outside the image bounds."""
        if not self.config.out_of_bound_removal:
            return landmarks

        filtered_landmarks = []
        for landmark in landmarks:
            # Convert normalized coordinates to pixel coordinates for bounds checking
            pixel_x = landmark.x * image_width
            pixel_y = landmark.y * image_height

            if 0 <= pixel_x <= image_width and 0 <= pixel_y <= image_height:
                filtered_landmarks.append(landmark)
            else:
                logger.debug(
                    f"Filtered out-of-bound landmark {landmark.landmark_id}: "
                    f"({pixel_x:.1f}, {pixel_y:.1f}) outside bounds ({image_width}, {image_height})"
                )

        logger.info(f"Filtered {len(landmarks) - len(filtered_landmarks)} out-of-bound landmarks")
        return filtered_landmarks

    def process_frame(self, frame_s3_uri: str) -> Tuple[str, Optional[ManifestItem], Optional[FaceLandmarkMetadata]]:
        """Process a single frame to extract face landmarks.

        Returns:
            Tuple of (status, manifest_item, metadata) where:
            - status: 'success', 'no_faces', or 'error'
            - manifest_item: ManifestItem for successful processing, None otherwise
            - metadata: FaceLandmarkMetadata for successful processing, None otherwise
        """
        try:
            bucket, s3_key = parse_s3_uri(frame_s3_uri)
            filename = Path(s3_key).name

            # Download frame from S3
            try:
                local_frame_path = self.s3_client.download_file_to_temp_dir(
                    bucket=bucket, s3_key=s3_key, temp_dir=self._temp_dir
                )
            except S3ClientError as e:
                logger.error(f"Failed to download frame {s3_key}: {e}")
                return "error", None, None

            # Load image
            image_bgr = cv2.imread(str(local_frame_path))
            if image_bgr is None:
                logger.error(f"Failed to load image: {local_frame_path}")
                return "error", None, None

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_height, image_width = image_rgb.shape[:2]

            # Create MediaPipe image and detect face landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.face_landmarker.detect(mp_image)

            annotated_s3_key = None
            annotated_local_path = None
            landmarks_data = {}
            num_faces = 0

            if detection_result.face_landmarks:
                num_faces = len(detection_result.face_landmarks)

                # Process landmarks for each face
                all_faces_landmarks = []
                for face_idx, face_landmarks in enumerate(detection_result.face_landmarks):
                    face_landmarks_list = []
                    for landmark_id, landmark in enumerate(face_landmarks):
                        face_landmark = FaceLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            landmark_id=landmark_id,
                            visibility=getattr(landmark, "visibility", None),
                            presence=getattr(landmark, "presence", None),
                        )
                        face_landmarks_list.append(face_landmark)

                    # Filter out-of-bound landmarks if enabled
                    face_landmarks_list = self._filter_out_of_bound_landmarks(
                        face_landmarks_list, image_width, image_height
                    )
                    all_faces_landmarks.append(face_landmarks_list)

                landmarks_data["faces"] = [
                    [landmark.model_dump() for landmark in face_landmarks] for face_landmarks in all_faces_landmarks
                ]

                # Process blend shapes if available
                if detection_result.face_blendshapes and self.config.output_face_blendshapes:
                    all_faces_blendshapes = []
                    for face_blendshapes in detection_result.face_blendshapes:
                        blendshapes_list = []
                        for blendshape in face_blendshapes:
                            face_blendshape = FaceBlendShape(
                                category_name=blendshape.category_name, score=blendshape.score
                            )
                            blendshapes_list.append(face_blendshape)
                        all_faces_blendshapes.append(blendshapes_list)
                    landmarks_data["blendshapes"] = [
                        [blendshape.model_dump() for blendshape in face_blendshapes]
                        for face_blendshapes in all_faces_blendshapes
                    ]

                # Process transformation matrices if available
                if (
                    detection_result.facial_transformation_matrixes
                    and self.config.output_facial_transformation_matrixes
                ):
                    matrices = []
                    for matrix in detection_result.facial_transformation_matrixes:
                        matrices.append(matrix.tolist() if hasattr(matrix, "tolist") else matrix)
                    landmarks_data["transformation_matrices"] = matrices

                # Save annotated image if enabled
                if self.config.save_annotated_images:
                    annotated_image = self._draw_landmarks_on_image(image_rgb, detection_result)
                    annotated_s3_key = f"{self._get_annotated_dir_key(s3_key)}{filename}"

                    # Save annotated image
                    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    annotated_local_path = self._temp_dir / f"annotated_{filename}"
                    cv2.imwrite(str(annotated_local_path), annotated_bgr)

                    self.s3_client.upload_file(
                        local_path=annotated_local_path,
                        bucket=self.s3_client.output_bucket,
                        s3_key=annotated_s3_key,
                    )

                # Save landmarks JSON
                landmarks_s3_key = f"{self._get_landmarks_dir_key(s3_key)}{Path(filename).stem}.json"
                self.s3_client.upload_json_as_dict(
                    data=landmarks_data,
                    bucket=self.s3_client.output_bucket,
                    s3_key=landmarks_s3_key,
                )

                # Create and save metadata
                metadata = FaceLandmarkMetadata(
                    frame_source_s3_key=s3_key,
                    landmarks_json_s3_key=landmarks_s3_key,
                    annotated_image_s3_key=annotated_s3_key,
                    num_faces=num_faces,
                    has_blendshapes=bool(detection_result.face_blendshapes and self.config.output_face_blendshapes),
                    has_transformation_matrix=bool(
                        detection_result.facial_transformation_matrixes
                        and self.config.output_facial_transformation_matrixes
                    ),
                )

                metadata_s3_key = f"{self._get_metadata_dir_key(s3_key)}{Path(filename).stem}.json"
                self.s3_client.upload_json_as_dict(
                    data=metadata.model_dump(mode="json"),
                    bucket=self.s3_client.output_bucket,
                    s3_key=metadata_s3_key,
                )

                # Create manifest item
                output_manifest_item = ManifestItem(
                    file_uri=f"s3://{self.s3_client.output_bucket}/{landmarks_s3_key}",
                    metadata_file=f"s3://{self.s3_client.output_bucket}/{metadata_s3_key}",
                    frame_source_s3_key=s3_key,
                )

                # Clean up local files
                local_frame_path.unlink(missing_ok=True)
                if annotated_local_path:
                    annotated_local_path.unlink(missing_ok=True)

                return "success", output_manifest_item, metadata

            else:
                logger.info(f"No faces detected in frame: {filename}")
                # Clean up local files
                local_frame_path.unlink(missing_ok=True)
                return "no_faces", None, None

        except Exception as e:
            logger.error("Failed to process frame %s: %s", s3_key, e)
            return "error", None, None
        finally:
            if local_frame_path and local_frame_path.exists():
                local_frame_path.unlink(missing_ok=True)

    def process_batch(self) -> BatchFaceLandmarkResult:
        """Process a batch of frames from the S3 manifest."""
        start_time = datetime.now(timezone.utc)

        if not self.config.force_write and self.s3_client.check_output_exists():
            raise FaceLandmarkExtractorError(
                f"Output directory already exists: {self.config.s3.output_uri}. " "Use force_write=True to overwrite."
            )

        try:
            # Load manifest
            manifest_bucket, manifest_key = parse_s3_uri(self.config.s3.input_uri)
            manifest_content = self.s3_client.download_file_to_bytes(
                bucket=manifest_bucket, s3_key=manifest_key
            ).decode("utf-8")
            manifest_items = [ManifestItem.model_validate_json(line) for line in manifest_content.strip().split("\n")]
        except (S3ClientError, ValueError) as e:
            raise FaceLandmarkExtractorError(
                f"Failed to load or parse manifest from {self.config.s3.input_uri}: {e}"
            ) from e

        try:
            total_frames = len(manifest_items)
            logger.info(f"Processing {total_frames} frames with {self.config.max_workers} workers")

            frame_results = []
            output_manifest_items = []
            successful_extractions = 0
            no_faces_detected = 0
            unreadable_frames = 0

            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all frame processing tasks
                future_to_uri = {
                    executor.submit(self.process_frame, item.file_uri): item.file_uri for item in manifest_items
                }

                # Collect results
                for future in as_completed(future_to_uri):
                    frame_uri = future_to_uri[future]
                    try:
                        status, manifest_item, metadata = future.result()
                        if status == "success":
                            successful_extractions += 1
                            if manifest_item:
                                output_manifest_items.append(manifest_item)
                            if metadata:
                                frame_results.append(metadata)
                        elif status == "no_faces":
                            no_faces_detected += 1
                        else:
                            unreadable_frames += 1
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_uri}: {e}")
                        unreadable_frames += 1

            logger.info(f"Batch processing completed: {successful_extractions}/{total_frames} successful")

            # Upload manifest file
            manifest_s3_key = None
            if output_manifest_items:
                manifest_content = "\n".join([item.model_dump_json() for item in output_manifest_items])
                manifest_s3_key = f"{self.s3_client.output_prefix}manifest.jsonl"
                self.s3_client.upload_data(
                    data=manifest_content.encode("utf-8"),
                    bucket=self.s3_client.output_bucket,
                    s3_key=manifest_s3_key,
                )

            end_time = datetime.now(timezone.utc)

            # Create and save batch result
            result = BatchFaceLandmarkResult(
                total_frames=total_frames,
                successful_extractions=successful_extractions,
                no_faces_detected=no_faces_detected,
                unreadable_frames=unreadable_frames,
                manifest_s3_key=manifest_s3_key,
                frame_results=frame_results,
                processing_start_time=start_time,
                processing_end_time=end_time,
            )

            # Save job metadata
            output_job_metadata_s3_key = f"{self.s3_client.output_prefix}job_metadata.json"
            self.s3_client.upload_json_as_dict(
                data=result.model_dump(mode="json"),
                bucket=self.s3_client.output_bucket,
                s3_key=output_job_metadata_s3_key,
            )

            return result

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            logger.error(f"Batch processing failed: {e}")
            raise FaceLandmarkExtractorError(f"Batch processing failed: {e}") from e
        finally:
            # Clean up temp directory
            if self._temp_dir.exists():
                for temp_file in self._temp_dir.iterdir():
                    if temp_file.is_file():
                        temp_file.unlink(missing_ok=True)
