# Add at least one tag from each grouping below.
# Each tag must be a plain value on its own line. Do not use prefixes such as domain:, source:, or model-release:.

# For HuggingFace Publishing/Discovery under Metadata: Please pick one from the following and delete what is not applicable

## Hugging Face Metadata
 
Add a YAML metadata block at the very top of the Hugging Face dataset `README.md`. Replace placeholder values, delete guidance comments, and keep each list item on its own line. Additional relevant tags may be included as plain values.
 
Add a YAML metadata block at the very top of the Hugging Face dataset `README.md`. These are the fields shown in the Hugging Face Metadata UI. <br>
 
```yaml
---

license:
# Use Hugging Face license identifiers. If more than one license applies, list each license on its own line.
- <license-id> # e.g. cc-by-4.0
- <additional-license-id> # optional, e.g. apache-2.0
 
license_name: <custom-license-name> # Required only when license is "other".
license_link: <custom-license-url> # Required only when license is "other".
 
# Choose One or More Allowed HuggingFace Task Slugs:
 
Domain: [agentic, code, content moderation, chat, finance, instruction following, math, medicine, parsing, personas, reasoning, safety, science, software engineering, science, tool use, vision-language, other, personas]
  
Required_Modality_Tags: [3d, audio, geospatial, image, tabular, text, timeseries, video]
  
Task_Categories: [Insert Task Category][XX%] [Insert Task Category][XX%] [Insert Task Category][XX%] if available
 
Natural_Language_Processing (NLP): <br>
Feature-Extraction, Fill-Mask, Masked-Language-Modeling, Multiple-Choice, Question-Answering, Sentence-Similarity, Summarization, Table-Question-Answering, Text-Classification, Text-Generation, Text-Ranking, Text-Retrieval, Text2text-Generation, Token-Classification, Translation, Zero-Shot-Classification
 
Audio: <br>
Audio-to-Audio, Audio-Classification, Audio-Question-Answering, Automatic-Speech-Recognition, Audio-Text-to-Speech, Audio-Text-to-Text, Text-to-Audio, Voice-Activity-Detection
 
Computer_Vision/Video:
Depth-Estimation, Document-Question-Answering, Image-Classification, Image-Feature-Extraction, Image-Segmentation, Mask-Generation, Object-Detection, Unconditional-Image-Generation, Visual-Question-Answering, Zero-Shot-Image-Classification, Zero-Shot-Object-Detection
 
Multimodal/Three_Dimensional (3D): <br>
Any-to-Any, Image-to-3D, Image-to-Image, Image-to-Text, Image-Text-to-Text, Image-to-Video, Text-to-3D, Text-to-Image, Text-to-Video, Video-Classification, Video-Text-to-Text, Visual-Document-Retrieval, Visual-Question-Answering
 
Tabular/Reinforcement_Learning/Other:
Reinforcement-Graph-ML, Learning, Robotics, Tabular-Classification, Tabular-Regression, Time-Series-Forecasting, [Insert Other Here]

Source_: [blend, crowdsourced, human, synthetic, nemo-data-designer, etc.]
# If Nemo Data Designer was used, include `nemo-data-designer`.
- <source>
  
Required_Training_Stage_Tags: [Benchmark, Evaluation, Pre-Training, Post-Training-Blends, Reward-Modeling, Reinforcement-Learning, Reward-Modeling, Supervised Fine-Tuning]

Size_Categories: [n<1K, 1K< n <10K, 10K< n <100K, 100K< n <1M, 1M< n <10M, 10M< n <100M, 100M< n <1B, 1B< n <10B, 10B<n<100B, 100B<n<1T, n>1T ]
  
License: [Insert License Type: CC-By-4.0, CC-By-NC-4.0, Apache-2.0, Openrail, MIT, Llama3, Other]
 
License_Name: [Required only when license is "other"]
 
License_Link: [Required only when license is "other"]

language:
# Use ISO language codes. If more than one language applies, list each language on its own line.
- <language-code> # e.g. en
- <additional-language-code> # optional
 
task_categories:
# Full list of usable Hugging Face task categories: https://huggingface.co/tasks
- <hf-task-category> # e.g. text-generation
- <additional-hf-task-category> # optional
 
Associated Model Tags if applicable:  [Insert link to model release here]

### Example
 
```yaml
# ---
# license:
# - cc-by-4.0
# - apache-2.0
# - mit
# language:
# - en
# task_categories:
# - text-generation
# tags:
# - text
# - code
# - tool-use
# - supervised-fine-tuning
# - blend
# - SWE
# - Nemotron_3_Ultra
# size_categories:
# - 100K<n<1M
# ---

## Dataset Description:  

[Insert Dataset Description] <br>
Please pick one of the following to insert at the end of your description:[This dataset is ready for commercial/non-commercial use.] OR [This dataset is for research and development only.] OR [This dataset is for demonstration purposes and not for production usage.]

## Dataset Owner(s): <br> 
[Insert Dataset Owner Name]

## Dataset Creation Date: <br> 
[Insert Dataset Creation/Modified Last Date]

## Versioning: <br>
[Insert Unique Version Name Here] <br>

Relationship to Previous Version(s): [Replacement | Extension] - [Briefly describe whether this version is a replacement of a previous release.  If there is no previous version, this section can be ignored.] <br>

## License/Terms of Use: <br> 
[Insert License Name Here](Insert Link to License here)

## Intended Usage: <br> 
[Insert Use Case/Application Description]

## Dataset Characterization: <br> 
** Data Collection Method <br>  
* [Automated] <br>  
* [Automatic/Sensors] <br>  
* [Human] <br>  
* [Synthetic] <br>  
* [Undisclosed] - [Only note Undisclosed if the data collection methods have not been disclosed. If we are able to ascertain the data collection method, please describe using one of the other methods noted above. <br>  
* [Hybrid: _______, _______] - [If you have multiple datasets, instead of listing a Data Collection Method for each dataset, you may use "Hybrid" and list the applicable data collection methods encompassing all datasets. For example, if you used three datasets for evaluation, all of which used Human, Synthetic, and Automated methods, you may list one sentence under this section stating "Hybrid: Human, Synthetic, Automated" for the Data Collection Method.] <br>  
* [Not Applicable] <br>
** Labeling Method <br>  
* [Automated] - [Think “Scraper”] <br>  
* [Automatic/Sensors] - [Machine-derived] <br>  
* [Human] <br>  
* [Synthetic] <br>  
* [Undisclosed] - [Only note Undisclosed if the data collection methods have not been disclosed. If we are able to ascertain the data collection method, please describe using one of the other methods noted above.  <br>  
* [Hybrid: _______, _______] - [If you have multiple datasets, instead of listing a Labeling Method for each dataset, you may use "Hybrid" and list the applicable labeling methods encompassing all datasets. For example, if you used three datasets for evaluation, all of which used Human, Synthetic, and Automated methods, you may list one sentence under this section stating "Hybrid: Human, Synthetic, Automated."]<br>  
* [Not Applicable] <br>

## Dataset Format: <br>  
[Insert Modality and Format Classification]

## Dataset Quantification: <br>  
[Insert Record Count- Note number of records (images, video, audio files, etcetera)]  
[Insert Feature Count- Note features present in record count above (e.g. tags)]  
[Insert Measurement of Total Data Storage]

## Reference(s): <br>
(Insert Paper or Public Repo Location):

## Ethical Considerations: <br> 
[Insert Name of Company] believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Please report quality, risk, security vulnerabilities or AI Concerns [insert name of link for follow-up](insert link for follow-up)].

This template was created by NVIDIA.

Please use this template provided in the repo as you see fit per the Creative Commons Zero (CCO) License, adopting as you see fit. If you have recommendations or questions, please file an issue. Stay tuned for regular updates to these templates from our team.
