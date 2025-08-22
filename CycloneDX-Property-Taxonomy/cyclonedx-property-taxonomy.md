# NVIDIA CycloneDX Property Taxonomy, v1.0.0

This is the NVIDIA property taxonomy for CycloneDX, enabling the inclusion of NVIDIA-specific ModelCard properties in CycloneDX BOM files.

This taxonomy is based on [**NVIDIA's ModelCard Template Structure**](https://github.com/NVIDIA/Trustworthy-AI/tree/main/Model%20Card%2B%2B%20Templates) for AI model documentation and extends the standard CycloneDX ModelCard schema to support NVIDIA's model metadata requirements.

An example implementation can be found in the [NVIDIA-Nemotron-Nano-9B-v2 Model Card](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2/modelcard).

For more information about CycloneDX property taxonomies, see the official [CycloneDX documentation](https://github.com/CycloneDX/cyclonedx-property-taxonomy/blob/main/README.md).

## `nvidia` Namespace Taxonomy

| Namespace | Description |
|-----------|-------------|
| **nvidia:model** | Namespace for core model metadata and classification properties |
| **nvidia:architecture** | Namespace for detailed model architecture and design properties |
| **nvidia:dataset** | Namespace for dataset-specific metadata and licensing properties |
| **nvidia:inference** | Namespace for inference engine and computational load properties |
| **nvidia:bias** | Namespace for bias measurement and mitigation properties |
| **nvidia:privacy** | Namespace for privacy protection and data handling properties |
| **nvidia:safety** | Namespace for safety, security, and risk management properties |
| **nvidia:explainability** | Namespace for model explainability and transparency properties |

---

## `nvidia:model`

| Property | Description |
|----------|-------------|
| **nvidia:model:deployment_geography** | Geographic regions where the model can be deployed; possible values include `Global`, `Asia-Pacific (APAC)`, `Europe, Middle East, and Africa (EMEA)`, `Latin America (LATAM)`, `North America (NAM)`, or specific countries. This property enables compliance with regional AI regulations and deployment restrictions. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard. |
| **nvidia:model:release_dates** | Release dates and URLs across different platforms including build.nvidia.com, GitHub, Hugging Face, NGC, and other platforms; format MM/DD/YYYY with associated URLs. This property tracks model availability across NVIDIA's distribution ecosystem. Only valid for use in ModelCard properties. MAY occur multiple times per ModelCard depending on the number of published locations.|
| **nvidia:model:references** | List of references, links to publications/papers/articles, associated works, and lineage. This property provides academic and technical context for model development. Only valid for use in ModelCard properties. MAY occur multiple times per ModelCard. |
| **nvidia:model:versions** | Multiple model version tracking with unique identifiers and descriptions differentiating each version. This property enables comprehensive version management for model lifecycles. Only valid for use in ModelCard properties. MAY occur multiple times per ModelCard. |

---

## `nvidia:architecture`

| Property | Description |
|----------|-------------|
| **nvidia:architecture:base_model** | Name of the base model if applicable, formatted as 'This model was developed based on [insert base model name]'. This property provides lineage information for derivative models. Only valid for use in ModelCard properties. MAY occur at most once per ModelCard. |
| **nvidia:architecture:design_choices** | Description of design choices related to initialization techniques, hyperparameter tuning, regularization techniques, model optimization, damping, and training parameters; recommended 100 words or less. This property documents technical decisions in model development. Only valid for use in ModelCard properties. MAY occur at most once per ModelCard. |
| **nvidia:architecture:gpai_size_limits** | Size and length limits for each input modality for GPAI (General Purpose AI) models. This property is essential for GPAI compliance and usage guidelines. Only valid for use in ModelCard properties when model classification requires GPAI documentation. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:architecture:runtime_engines** | Supported runtime engines with versions; possible values include `BioNeMo`, `DeepStream`, `DXIS`, `Maxine`, `Morpheus`, `NeMo`, `Riva`, `TAO`, `Not Applicable`. This property specifies compatible inference frameworks. Only valid for use in NVIDIA ModelCard properties. MAY occur multiple times per NVIDIA ModelCard. |
| **nvidia:architecture:hardware_compatibility** | Supported hardware microarchitectures; possible values include `NVIDIA Ampere`, `NVIDIA Blackwell`, `NVIDIA Jetson`, `NVIDIA Hopper`, `NVIDIA Lovelace`, `NVIDIA Pascal`, `NVIDIA Turing`, `NVIDIA Volta`, or specific model names. This property defines hardware requirements for optimal performance. Only valid for use in NVIDIA ModelCard properties. MAY occur multiple times per NVIDIA ModelCard. |
| **nvidia:architecture:operating_systems** | Supported operating systems; possible values include `Linux`, `Linux 4 Tegra`, `QNX`, `Windows`, or other systems not listed. This property specifies platform compatibility requirements. Only valid for use in NVIDIA ModelCard properties. MAY occur multiple times per NVIDIA ModelCard. |

---

## `nvidia:dataset`

| Property | Description |
|----------|-------------|
| **nvidia:dataset:training_license** | Training dataset license information including name, link to license, or applicable Jira ticket/NVBug; internal-only field. This property ensures legal compliance for training data usage. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard with training data. |
| **nvidia:dataset:testing_license** | Testing dataset license information including name, link to license, or applicable Jira ticket/NVBug; internal-only field. This property ensures legal compliance for testing data usage. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard with testing data. |
| **nvidia:dataset:evaluation_license** | Evaluation dataset license information including name, link to license, or applicable Jira ticket/NVBug; internal-only field. This property ensures legal compliance for evaluation data usage. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard with evaluation data. |
| **nvidia:dataset:gpai:total_size** | Total size of GPAI dataset in number of data points, reported to at least two significant figures. This property quantifies the scale of training data for regulatory reporting. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:total_datasets** | Total number of datasets used in GPAI model training. This property provides dataset diversity metrics for compliance documentation. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:training_percentage** | Percentage of data used for training in GPAI model development. This property documents data allocation for transparency. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:testing_percentage** | Percentage of data used for testing in GPAI model development. This property documents data allocation for transparency. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:validation_percentage** | Percentage of data used for validation in GPAI model development. This property documents data allocation for transparency. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:training_period** | Time period for training data collection in GPAI model development. This property provides temporal context for training data. Only valid for use in GPAI ModelCard properties. SHOULD occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:testing_period** | Time period for testing data collection in GPAI model development. This property provides temporal context for testing data. Only valid for use in GPAI ModelCard properties. SHOULD occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:validation_period** | Time period for validation data collection in GPAI model development. This property provides temporal context for validation data. Only valid for use in GPAI ModelCard properties. SHOULD occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:data_processing** | Description of data processing methods used to transform acquired data into training data for GPAI models; recommended 200 words. This property documents data transformation processes for regulatory compliance. Only valid for use in GPAI ModelCard properties. SHOULD occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:harmful_data_methods** | Description of methods implemented in data acquisition or processing to address potentially harmful data in GPAI training, testing, or evaluation data; recommended 300 words. This property documents content safety measures for regulatory compliance. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:dataset:gpai:public_datasets** | List of main/large public datasets (above 5% of overall data) with unique identification, links, and collection periods used in GPAI model training. This property provides public data transparency. Only valid for use in GPAI ModelCard properties. MAY occur multiple times per GPAI ModelCard. |
| **nvidia:dataset:gpai:private_datasets** | List of main/large private datasets acquired from third parties (above 5% of overall data) with unique identifiers, links, and collection periods used in GPAI model training. This property provides private data transparency. Only valid for use in GPAI ModelCard properties. MAY occur multiple times per GPAI ModelCard. |
| **nvidia:dataset:web_crawled_size_per_modality** | Overall size per modality and period of scraping for web-crawled data used in model training. This property quantifies web-scraped content by type. Only valid for use in ModelCard properties using web-crawled data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:web_crawler_name** | Name of the web crawler along with purpose and behavior description. This property identifies web scraping tools and methods. Only valid for use in ModelCard properties using web-crawled data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:web_crawler_period** | Period of data collection for web crawler operations. This property provides temporal context for web scraping activities. Only valid for use in ModelCard properties using web-crawled data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:web_crawler_organization** | Organization operating the web crawler for data collection. This property identifies responsible parties for web scraping. Only valid for use in ModelCard properties using web-crawled data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:web_crawler_robots_txt** | Description of how crawler respects robots.txt preferences during data collection. This property documents web scraping compliance measures. Only valid for use in ModelCard properties using web-crawled data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:web_crawler_targeted_content** | Content targeted by the web crawler during data collection operations. This property describes the scope of web scraping activities. Only valid for use in ModelCard properties using web-crawled data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:web_top_domains** | List of top 10% of all internet domain names per type of data modality (e.g., text, image) for web-crawled data. This property provides source transparency for web-scraped content. Only valid for use in ModelCard properties using web-crawled data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:web_rights_respect_measures** | Measures implemented to respect reservations of rights from text and data-mining exception during data collection, including opt-out protocols and solutions honored. This property documents rights compliance for web scraping. Only valid for use in ModelCard properties using web-crawled data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:user_sourced_size_per_modality** | Overall size per modality for user-sourced data collected by the provider including prompts. This property quantifies user-generated content by type. Only valid for use in ModelCard properties using user-sourced data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:user_sourced_provider_services** | List of providers' services/products where user-sourced data was collected. This property identifies sources of user-generated content. Only valid for use in ModelCard properties using user-sourced data. MAY occur multiple times per relevant ModelCard. |
| **nvidia:dataset:synthetic_size_per_modality** | Overall size per modality for self-sourced synthetic datasets used in model training. This property quantifies synthetic content by type. Only valid for use in ModelCard properties using synthetic data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:synthetic_generation_methods** | Description of methods used to synthetically generate training data. This property documents artificial data creation techniques. Only valid for use in ModelCard properties using synthetic data. SHOULD occur exactly once per relevant ModelCard. |
| **nvidia:dataset:synthetic_ai_models_used** | Names of AI models or systems used to synthetically generate training data. This property identifies tools used for synthetic data creation. Only valid for use in ModelCard properties using synthetic data. MAY occur multiple times per relevant ModelCard. |

---

## `nvidia:inference`

| Property | Description |
|----------|-------------|
| **nvidia:inference:engine** | Inference engine specification; possible values include `TensorRT`, `Triton`, or `Other`. This property specifies the recommended inference runtime. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard. |
| **nvidia:inference:test_hardware** | List of specific test hardware models used for validation and performance testing. This property provides hardware compatibility and performance context. Only valid for use in ModelCard properties. MAY occur multiple times per ModelCard. |
| **nvidia:inference:additional_content** | Additional inference-related content that may be included for comprehensive documentation. This property allows for extensible inference metadata. Only valid for use in ModelCard properties. MAY occur at most once per ModelCard. |
| **nvidia:inference:hardware_units_count** | Number of hardware units used for training. This property quantifies training infrastructure scale for reproducibility. Only valid for use in ModelCard properties with internal computational load documentation. SHOULD occur exactly once per internal ModelCard. |
| **nvidia:inference:hardware_units_types** | Types of hardware units used for training (e.g., specific GPU models, TPUs). This property documents training infrastructure for reproducibility. Only valid for use in ModelCard properties with internal computational load documentation. MAY occur multiple times per internal ModelCard. |
| **nvidia:inference:training_duration_wall_clock** | Model training duration measured in wall clock time, reported in units of days. This property provides real-world training time metrics. Only valid for use in ModelCard properties with internal computational load documentation. SHOULD occur exactly once per internal ModelCard. |
| **nvidia:inference:training_duration_hardware** | Model training duration measured in hardware time, reported in units of hardware hours (e.g., GPU hours). This property provides hardware resource consumption metrics. Only valid for use in ModelCard properties with internal computational load documentation. SHOULD occur exactly once per internal ModelCard. |
| **nvidia:inference:gpu_utilization** | GPU utilization rate during model training. This property provides efficiency metrics for training operations. Only valid for use in ModelCard properties with internal computational load documentation. SHOULD occur exactly once per internal ModelCard. |
| **nvidia:inference:compute_operations** | The compute used during model training reported in units of integer or floating-point operations. This property quantifies computational requirements for model training. Only valid for use in ModelCard properties with internal computational load documentation. SHOULD occur exactly once per internal ModelCard. |

---

## `nvidia:bias`

| Property | Description |
|----------|-------------|
| **nvidia:bias:gpai_data_acquisition_processing** | For GPAI models, description of methods implemented in data acquisition or processing to address the prevalence of identifiable biases in the training, testing, and validation data; recommended 200 words. This property documents bias mitigation strategies for regulatory compliance. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |

---

## `nvidia:privacy`

| Property | Description |
|----------|-------------|
| **nvidia:privacy:dataset_review_frequency** | Description of how often the dataset is reviewed for privacy compliance and data quality. This property establishes ongoing privacy governance practices. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard using personal data. |
| **nvidia:privacy:privacy_policy** | Reference to the applicable privacy policy governing the model and its data usage. This property provides legal framework reference for data handling. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard. |
| **nvidia:privacy:gpai_copyrighted_materials** | For GPAI models, description of methods implemented in data acquisition or processing to address the prevalence of copyrighted materials in training, testing, and validation data. This property documents copyright compliance measures. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:privacy:gpai_rightsholder_content** | For GPAI models, measures implemented after data collection to identify and remove content for which rights have been reserved by rightsholders. This property documents post-collection content filtering for rights compliance. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |

---

## `nvidia:safety`

| Property | Description |
|----------|-------------|
| **nvidia:safety:gpai_data_acquisition_processing** | Description of any methods implemented in data acquisition or processing, if any, to address illegal or harmful content in the training data, including, but not limited to, child sexual abuse material (CSAM) and non-consensual intimate imagery (NCII). Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:safety:life_critical_impact** | Describe the life critical impact of the application. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard when applicable. |
| **nvidia:safety:model_application** | Model Application Field(s): List where model could be reasonably integrated: Select from the following: Autonomous Vehicles OR Aviation OR Climate OR Critical infrastructure OR Customer Service OR Healthcare OR Industrial/Machinery and Robotics OR Media & Entertainment OR [Insert Name Relevant Field Here]. This property enables AI Act compliance and risk assessment. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard when applicable. |
| **nvidia:safety:use_case_restrictions** | Provide the name of approved license agreement and reference link to the agreement. Only valid for use in ModelCard properties. MUST occur exactly once per ModelCard. |
| **nvidia:safety:model_dataset_restrictions** | Description of model and dataset access restrictions; includes principle of least privilege (PoLP) application, dataset access limitations during training, and dataset license constraint adherence. This property documents security and access controls. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard. |
| **nvidia:safety:gpai_harmful_data_methods** | Description of methods implemented in data acquisition or processing, if any, to address other types of potentially harmful data in the training, testing, and validation data. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |
| **nvidia:safety:gpai_csam_ncii_removal** | For GPAI models, description of methods implemented in data acquisition or processing to address illegal or harmful content in the training data, including but not limited to child sexual abuse material (CSAM) and non-consensual intimate imagery (NCII). This property documents critical content safety measures. Only valid for use in GPAI ModelCard properties. MUST occur exactly once per GPAI ModelCard. |

---

## `nvidia:explainability`

| Property | Description |
|----------|-------------|
| **nvidia:explainability:synthetic_data_evaluation** | For GPAI models, tools used to evaluate datasets to identify synthetic data and ensure data authenticity. This property documents data verification and authenticity measures. Only valid for use in GPAI ModelCard properties. SHOULD occur exactly once per GPAI ModelCard using synthetic data. |
| **nvidia:explainability:quality_standards** | Description of NVIDIA quality standards that have been verified and met by the model. This property documents internal quality assurance compliance. Only valid for use in NVIDIA ModelCard properties. SHOULD occur exactly once per NVIDIA ModelCard. |
| **nvidia:explainability:model_workings** | Description of how the model works, providing technical insight into model operation and decision-making processes. This property enhances model transparency and interpretability. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard. |
| **nvidia:explainability:terms_of_use** | Description of terms of use and licensing information specific to model explainability and interpretation tools. This property governs the usage of explainability features. Only valid for use in ModelCard properties. SHOULD occur exactly once per ModelCard with explainability features. |

---

## Contacts

For questions or contributions to this taxonomy, please reach out to [Pratyusha Maiti](mailto:pmaiti@nvidia.com).
