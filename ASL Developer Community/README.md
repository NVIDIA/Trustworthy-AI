# ASL Data Pipeline

A comprehensive data processing pipeline for American Sign Language (ASL) data, designed for extracting and processing key frames, hand landmarks, and pose landmarks from ASL videos. This pipeline is built with production-grade features including S3 integration, parallel processing, and robust error handling.

## 🚀 Features

- **Multi-stage Processing Pipeline**: 4-step automated pipeline for ASL video processing
- **Key Frame Extraction**: Intelligent motion analysis to extract meaningful frames from videos
- **Hand Landmark Detection**: MediaPipe-based hand landmark extraction with high accuracy  
- **Pose Landmark Detection**: Full body pose landmark extraction for comprehensive analysis
- **Face Landmark Detection**: Optional face mesh extraction with 468+ landmarks and blend shapes
- **SuperAnnotate Format Conversion**: Convert extracted data to SuperAnnotate format for annotation workflows
- **S3 Integration**: Seamless AWS S3 input/output with support for both folders and manifest files

## 📋 Table of Contents

- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.11 (Required - MediaPipe compatibility)
- Poetry (for dependency management)
- AWS CLI configured (for S3 operations)

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/yourusername/asl-data-pipeline.git
cd asl-data-pipeline

# Install dependencies
poetry install

# Install with development dependencies
poetry install --with dev
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_PROFILE=your-aws-profile
AWS_DEFAULT_REGION=us-east-1

# Optional: Direct AWS credentials
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
```

## 📊 Pipeline Overview

The ASL Data Pipeline processes video data through four main stages:

```
Video Files (S3) → Key Frame Extraction → Hand Landmark Extraction → Pose Extraction → SuperAnnotate Format Conversion
```

### Pipeline Flow

1. **Key Frame Extraction**: Analyzes video motion to extract meaningful frames
2. **Hand Landmark Extraction**: Detects hand landmarks in extracted frames  
3. **Pose Extraction**: Extracts full body pose landmarks
4. **SuperAnnotate Format Conversion**: Converts landmarks to SuperAnnotate format for annotation workflows

### Optional Features

- **Face Landmark Extraction**: Extract detailed face mesh with 468+ landmarks, blend shapes, and facial transformation matrices (available as separate module)

## ⚡ Quick Start

### Using the Jupyter Notebook

The fastest way to get started is using the provided notebook:

```bash
# Activate the poetry environment
poetry shell

# (Optional) Create a local copy of the notebook for experimentation
cp notebooks/asl_data_pipeline.ipynb notebooks/asl_data_pipeline_local.ipynb

# Start Jupyter
jupyter notebook

# Open notebooks/asl_data_pipeline.ipynb (or your local copy)
```

**Tip:** If you create local notebook copies for experimentation, add your naming pattern to your global gitignore to prevent accidentally committing them:

```bash
# Add to your global gitignore (one-time setup)
echo "*_local.ipynb" >> ~/.gitignore_global
git config --global core.excludesfile ~/.gitignore_global
```

### Python Script Usage

```python
from asl_data_pipeline.models.s3 import InputType
from asl_data_pipeline.preprocessing.key_frame_extractor.extractor import (
    KeyFrameExtractor, 
    KeyFrameExtractorConfig
)
from asl_data_pipeline.preprocessing.key_frame_extractor.models import S3Config

# Configure the pipeline
config = KeyFrameExtractorConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/videos/",
        output_uri="s3://your-bucket/keyframes/",
        aws_profile="your-aws-profile",
        input_type=InputType.FOLDER,
    ),
    max_videos=4,
    force_write=True,
)

# Run key frame extraction
extractor = KeyFrameExtractor(config)
results = extractor.process_videos_batch()

print(f"Processed {results.successful_extractions} videos successfully")
```

## 🔄 Pipeline Steps

### Step 1: Key Frame Extraction

Extracts meaningful frames from videos using motion analysis:

```python
key_frame_extractor_config = KeyFrameExtractorConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/videos/",
        output_uri="s3://your-bucket/keyframes/",
        aws_profile="your-aws-profile",
        input_type=InputType.FOLDER,
    ),
    max_videos=4,
    force_write=True,
)
frame_extractor = KeyFrameExtractor(key_frame_extractor_config)
frame_extractor.process_videos_batch()
```

**Key Features:**
- Motion analysis using optical flow
- Active region detection
- Sharpness refinement for frame selection
- Automatic generation of processing manifest

### Step 2: Hand Landmark Extraction

Detects hand landmarks using MediaPipe:

```python
hand_landmark_extractor_config = HandLandmarkExtractorConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/keyframes/manifest.jsonl",
        output_uri="s3://your-bucket/hand_landmarks/",
        aws_profile="your-aws-profile", 
        input_type=InputType.MANIFEST,
    ),
    max_workers=4,
    force_write=True,
)
hand_landmark_extractor = HandLandmarkExtractor(hand_landmark_extractor_config)
hand_landmark_extractor.process_batch()
```

**Features:**
- 21 hand landmarks per hand
- Confidence scoring
- Both left and right hand detection
- Robust handling of partial occlusions

### Step 3: Pose Extraction

Extracts full body pose landmarks:

```python
pose_extractor_config = PoseExtractorConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/keyframes/manifest.jsonl",
        output_uri="s3://your-bucket/pose_landmarks/",
        aws_profile="your-aws-profile",
        input_type=InputType.MANIFEST,
    ),
    out_of_bound_removal=True,
    max_workers=4,
    force_write=True,
)
pose_extractor = PoseExtractor(pose_extractor_config)
pose_extractor.process_batch()
```

**Features:**
- 33 pose landmarks
- Out-of-bound coordinate removal
- Visibility and presence scoring
- Upper body focus for ASL analysis

### Step 4: SuperAnnotate Format Conversion

Converts landmark data to SuperAnnotate format for annotation workflows:

```python
# Hand landmarks conversion
hand_sa_converter_config = HandSAConverterConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/hand_landmarks/manifest.jsonl",
        output_uri="s3://your-bucket/hand_landmarks_sa/",
        aws_profile="your-aws-profile",
        input_type=InputType.MANIFEST,
    ),
    force_write=True,
    max_workers=4,
)
hand_sa_converter = HandSAConverter(hand_sa_converter_config)
hand_sa_converter.process_batch()

# Pose landmarks conversion  
pose_sa_converter_config = PoseSAConverterConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/pose_landmarks/manifest.jsonl",
        output_uri="s3://your-bucket/pose_landmarks_sa/",
        aws_profile="your-aws-profile", 
        input_type=InputType.MANIFEST,
    ),
    force_write=True,
    max_workers=4,
)
pose_sa_converter = PoseSAConverter(pose_sa_converter_config)
pose_sa_converter.process_batch()
```

## 🎭 Optional: Face Landmark Extraction

For applications requiring detailed facial analysis, an optional face landmark extraction module is available:

### Features
- **468+ Face Landmarks**: Full face mesh with detailed landmark positions
- **Blend Shapes**: 52 facial expression coefficients for animation
- **Transformation Matrices**: 3D facial pose and orientation data
- **Multiple Face Support**: Detect and track multiple faces simultaneously

### Usage

```python
from asl_data_pipeline.preprocessing.face_landmark_extractor import (
    FaceLandmarkExtractor,
    FaceLandmarkExtractorConfig,
)

face_config = FaceLandmarkExtractorConfig(
    s3=S3Config(
        input_uri="s3://your-bucket/keyframes/manifest.jsonl",
        output_uri="s3://your-bucket/face_landmarks/",
        aws_profile="your-aws-profile",
        input_type=InputType.MANIFEST,
    ),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    max_workers=4,
    force_write=True,
)

face_extractor = FaceLandmarkExtractor(face_config)
face_results = face_extractor.process_batch()
```

## ⚙️ Configuration

### S3 Configuration

All pipeline steps use the same S3 configuration pattern:

```python
from asl_data_pipeline.models.s3 import S3Config, InputType

s3_config = S3Config(
    input_uri="s3://bucket/path/",           # Required: Input S3 URI
    output_uri="s3://bucket/output/",        # Required: Output S3 URI  
    aws_profile="your-profile",              # Optional: AWS profile
    region_name="us-east-1",                 # Optional: AWS region
    input_type=InputType.FOLDER,             # FOLDER or MANIFEST
)
```

### Processing Parameters

Each processor supports customization:

```python
# Key frame extraction parameters
key_frame_config = KeyFrameExtractorConfig(
    s3=s3_config,
    max_videos=10,           # Limit number of videos to process (useful for testing)
    max_workers=4,           # Number of parallel processing threads
    force_write=True,        # Overwrite existing output files
    temp_dir=Path("/tmp"),   # Temporary processing directory
    enable_charts=True,      # Generate motion analysis charts
)

# Hand/Pose extraction parameters
extractor_config = HandLandmarkExtractorConfig(
    s3=s3_config,
    max_workers=4,           # Number of parallel processing threads
    force_write=True,        # Overwrite existing output files
    temp_dir=Path("/tmp"),   # Temporary processing directory
)
```

#### Key Parameters Explained

**`max_videos`** (Key Frame Extractor only)
- Limits the number of videos to process from the input source
- Useful for testing the pipeline on a subset of data before processing entire datasets
- If not specified or set to `None`, processes all videos in the input location

**`max_workers`** (All extractors and converters)
- Controls the number of parallel processing threads
- Recommended values: 2-8 depending on your system resources
- Use lower values (2-4) for memory-intensive operations or limited resources

## 📁 Project Structure

```
asl-data-pipeline/
├── src/asl_data_pipeline/
│   ├── models/                          # Data models
│   │   ├── s3.py                       # S3 configuration models
│   │   └── manifest.py                 # Manifest data models
│   ├── preprocessing/                   # Core processing modules
│   │   ├── key_frame_extractor/        # Video key frame extraction
│   │   ├── hand_landmark_extractor/    # Hand landmark detection
│   │   ├── pose_extractor/             # Pose landmark detection
│   │   ├── face_landmark_extractor/    # Face landmark detection (optional)  
│   │   ├── hand_sa_converter/          # Hand SA format conversion
│   │   └── pose_sa_converter/          # Pose SA format conversion
│   └── utils/                          # Utility functions
│       ├── s3_client.py               # S3 operations
│       ├── s3_utils.py                # S3 utilities
│       ├── video_utils.py             # Video processing
│       └── image_processing.py        # Image utilities
├── notebooks/
│   └── asl_data_pipeline.ipynb        # Main processing notebook
├── tests/                             # Test suite
├── data/                             # Data directories
├── docs/                             # Documentation
└── scripts/                          # Example scripts
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install with development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests with coverage
make test

# Clean temporary files
make clean
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run coverage run -m pytest
poetry run coverage report -m

# Run specific test
poetry run pytest tests/preprocessing/test_key_frame_extractor.py
```

## 📖 Usage Examples

### Complete Pipeline Example

See the main notebook `notebooks/asl_data_pipeline.ipynb` for a complete pipeline example processing ASL videos through all stages.

### Individual Component Usage

Each processing component can be used independently:

```python
# Just extract key frames
from asl_data_pipeline.preprocessing.key_frame_extractor import KeyFrameExtractor

extractor = KeyFrameExtractor(config)
results = extractor.process_videos_batch()

# Process specific videos
specific_videos = ["video1.mp4", "video2.mp4"]  
results = extractor.process_videos_batch(video_s3_keys=specific_videos)
```

### Error Handling

```python
try:
    results = extractor.process_videos_batch()
    print(f"Success: {results.successful_extractions}")
    print(f"Failed: {results.failed_extractions}")
    
    for error in results.errors:
        print(f"Error processing {error.video_s3_key}: {error.error_message}")
        
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
```

### Data Merger (External Annotation Workflow) 

**Note:** This is a standalone utility separate from the main pipeline notebook. 
 
 Use this when you have body pose, face, and hand annotations from external sources (e.g., after manual annotation in SuperAnnotate) that need to be **merged into a single JSON**.

**Requirements:** 
- Python 3.6+  
- Input structure:
```
.
├── Body Pose/
│   ├── *_keyframe_004*.json
│   ├── *_keyframe_005*.json
│   └── ...
├── Face/
│   ├── *_keyframe_004*.json
│   ├── *_keyframe_005*.json
│   └── ...
└── Hand/
    ├── *_keyframe_004*.json
    ├── *_keyframe_005*.json
    └── ...
```

**Usage:**

1. Create a `merge.py` file and copy the following code:
```python
import json, glob, re, os
from pathlib import Path
from collections import defaultdict

files = defaultdict(dict)
for folder, label in [('Body Pose', 'body'), ('Face', 'face'), ('Hand', 'hand')]:
    for f in glob.glob(f'{folder}/*.json*'):
        key = re.search(r'(.*_keyframe_\d+)', Path(f).name).group(1)
        files[key][label] = json.load(open(f))

os.makedirs('merged', exist_ok=True)
for key, data in files.items():
    if len(data) == 3:
        json.dump(data, open(f'merged/{key}.json', 'w'), indent=2)
```

2. On your terminal, run: `python3 merge.py`  

**Output:** 
`merged/` directory with merged files structured as `{body: {...}, face: {...}, hand: {...}}`  

⚠️ No validation performed—assumes well-formed input files.


## 🔍 Monitoring and Debugging

### Logging

Enable detailed logging:

```python
import logging
logging.getLogger("asl_data_pipeline").setLevel(logging.DEBUG)
```

### Output Structure

Each processing step creates structured output:

```
s3://bucket/output/
├── video_name/
│   ├── artifacts/           # Processed files (frames, landmarks)
│   ├── metadata.json       # Processing metadata
│   └── charts/             # Analysis charts (if enabled)
├── manifest.jsonl          # Processing manifest
└── job_metadata.json       # Batch processing summary
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed
- Use Pydantic models for data validation

## 📝 License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for landmark detection capabilities
- [OpenCV](https://opencv.org/) for video processing
- [SuperAnnotate](https://www.superannotate.com/) for annotation platform integration
- ASL community for domain expertise and validation

Built with ❤️ for the ASL community
