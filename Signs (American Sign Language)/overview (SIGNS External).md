# Model Overview


### Description:
SIGNS detects and classifies American Sign Language hand gestures in real-time video streams using MediaPipe Holistic landmark detection combined with custom template matching algorithms.  <br>


### License/Terms of Use
This model is not approved for external distribution.  No audio or video data is captured for learning; Abide by the [data collection terms](https://signs-ai.com/datacollectionterms) for contributions.  <br>

### Deployment Geography:
Global - web-based application accessible worldwide <br>

### Use Case: <br>
Consumers learning American Sign Language <br>

### Release Date:  <br>
February 2025: signs-ai.com <br> 


## References(s):
* [MediaPipe Holistic] (https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) <br> 
* [The Cross-linguistic Distribution of Sign Language Parameters] (https://escholarship.org/uc/item/1kz4f6q7) <br>

## Model Architecture:
**Architecture Type:** Hybrid Computer Vision Pipeline <br>
**Network Architecture:** MediaPipe Holistic + Custom Template Matching + State Machine <br>


## Input: <br>
**Input Type(s):** Video <br>
**Input Format:** Red, Green, Blue (RGB) Video <br>
**Input Parameters:** Two Dimensional (2D) video frames <br>
**Other Properties Related to Input:** Minimum 480x360 resolution, 15-30fps recommended, requires clear view of hands and upper body, adequate lighting conditions, single person in frame optimal <br>

## Output: <br>
**Output Type(s):** Text, Visual Cues to Affirm and Correct American Sign Language parameters <br>
**Output Format:** JSON/String, Video (Animation) <br>
**Output Parameters:** One-Dimensional (1D) Structured Data <br>

The system is designed to run on standard web browsers with WebGL support and can leverage GPU acceleration where available. <br>

## Software Integration:
**Runtime Engine(s):** 
* MediaPipe (via CDN or local installation) <br>
* Web Browser (JavaScript/WebGL) <br>


**Supported Hardware Microarchitecture Compatibility [List in Alphabetic Order]:** <br>
* GPU acceleration via WebGL-compatibility <br>
* Web browser environment <br>

**Preferred/Supported Operating System(s):**
* [Linux] <br>
* [Windows] <br>
* [macOs] <br>
* Mobile(iOS/Android via browser) <br>

## Model Version(s):
v1.0-dev  <br>


## Training, Testing, and Evaluation Datasets:

## Training Dataset: No training data was used in developing this model; it is based purely on a static, numerical representation of single instances of words and short phrases and does not account for variations (including regional and differences in ethnic and regional dialects). 
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Not Applicable (N/A) <br>
**Dataset License(s):** N/A <br>


### Evaluation Dataset:
No training data was used in developing this model; it is based purely on a static, numerical representation of single instances of words and short phrases and does not account for variations (including regional and differences in ethnic and regional dialects). 
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Not Applicable (N/A) <br>
**Dataset License(s):** N/A <br>

# Inference:
**Acceleration Engine:**  Web Browser JavaScript Engine, WebGL (where available) <br>
**Test Hardware:** <br>  
* Standard laptops/desktops with webcams <br>
* Modern web browsers (Chrome, Firefox, Safari, Edge) <br>
* Minimum: Dual-core CPU, 4GB RAM, integrated graphics with WebGL support <br>

**Performance Characteristics:** <br>  
* Real-time processing: 15-30 FPS <br>
* Latency: <100ms for sign detection <br>
* Memory usage: <500MB browser memory <br>

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
