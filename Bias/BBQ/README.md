# BBQ Eval

This version of BBQ enables OpenAI-compatible endpoints, automates metric review, and extends the method to audio inputs.

## Acknowledgments

A special thank you to **Alicia Parrish**, lead author of BBQ, and to **Google** for their support in bringing this work to NVIDIA's Trustworthy AI repo.

BBQ (Bias Benchmark for QA) is a hand-built dataset that tests how social biases show up in model answers. It covers nine social dimensions relevant to U.S. English-speaking contexts. BBQ has become a widely used standard for measuring stereotypical bias in language models, and it plays a direct role in how NVIDIA evaluates and documents bias in its models. We're grateful to the BBQ authors for releasing it openly.

**Original repository:** [github.com/nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)

If you use BBQ, please cite the original paper:
> Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. 2022. [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://aclanthology.org/2022.findings-acl.165/). In *Findings of the Association for Computational Linguistics: ACL 2022*.

Evaluate language-model bias with the [Bias Benchmark for Question Answering
(BBQ)](https://github.com/nyu-mll/BBQ). BBQ Eval generates responses through
OpenAI-compatible APIs, extracts answer choices, and produces category-level
and overall bias metrics. Text subsets and an audio dataset are included.

## Installation

Requires **Python 3.15**. Clone the repository and install from `Bias/BBQ`:

```bash
git clone https://github.com/NVIDIA/Trustworthy-AI.git
cd Trustworthy-AI/Bias/BBQ

python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start

Set an API key for the endpoint in the
[default configuration](bbq_eval/openai/default_config.yaml), then run a small
text evaluation:

```bash
export API_KEY='your-api-key'
bbq_eval --small_dataset --num_rows 10 --output_dir results/smoke
```

Run the complete bundled text subset with checkpoint recovery:

```bash
bbq_eval --small_dataset --output_dir results/model --resume
```

For a different model or endpoint, copy the default configuration to
`model.local.yaml`, update `model.model_name` and `model.base_url`, and pass
`--config model.local.yaml`. Keep API keys in environment variables.

Use `--categories Age Religion` to select categories, `--dataset path/to/data.json`
to load local JSON or JSON Lines, or `bbq_eval --help` for all options.
Without `--small_dataset` or `--dataset`, the CLI loads `heegyu/bbq` from
Hugging Face. API usage may incur charges.

## Results

Files are written to the selected output directory:

| File | Contents |
| --- | --- |
| `responses_optimized_*.json` | Model responses and extracted answers |
| `bbq_metrics.csv` | Ambiguous-context bias scores by category |
| `bbq_metrics_overall.csv` | Pooled ambiguous-context bias score |
| `checkpoints/*.jsonl` | Saved attempts for resuming a run |

`--resume` skips successful examples and retries failed extractions. Keep the
same dataset and configuration when resuming. Failed extractions are excluded
from metrics; review their reported count alongside the scores. For ambiguous
contexts, the calculation includes unknown answers in its denominator, a
variation from the original BBQ analysis.

## Audio dataset

The [audio dataset](bbq_eval/data/audio-bbq-dataset) contains a JSON Lines manifest
with **3,672 examples** and WAV recordings stored through Git LFS. To download
the audio, install Git LFS and run:

```bash
git lfs install
git lfs pull --include="Bias/BBQ/bbq_eval/data/audio-bbq-dataset/**"
```

Each manifest entry's `audio_file` path is relative to the manifest directory.
The evaluation CLI processes text; use the WAV files with an audio-capable
inference pipeline. Audio files are distributed through the repository rather
than the Python package.

## Development

```bash
python -m pip install pytest pytest-asyncio ruff
python -m pytest
python -m ruff check bbq_eval tests
```
