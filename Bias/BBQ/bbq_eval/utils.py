"""Shared configuration, dataset and JSON handling."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parent


def load_model_config(config_file):
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, dict) or not model.get("model_name"):
        raise ValueError("Configuration must contain model.model_name")
    return model


def resolve_api_key(config, parser=False):
    """Prefer environment credentials; a separate parser key is optional."""
    if parser:
        key = os.getenv("PARSER_API_KEY") or config.get("parser_api_key")
        if key:
            return key
    key = (
        os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or config.get("api_key")
    )
    if not key:
        raise ValueError("Set API_KEY, OPENAI_API_KEY or NVIDIA_API_KEY")
    return key


def load_category(category, dataset):
    """Resolve local paths from the caller, bundled names from the package."""
    path = Path(dataset).expanduser()
    if not path.is_file() and path.name == str(dataset):
        path = PACKAGE_ROOT / "data" / dataset
    if path.is_file():
        # Existing small datasets include both JSON arrays and JSON Lines.
        with path.open(encoding="utf-8") as stream:
            first = next((c for c in iter(lambda: stream.read(1), "") if not c.isspace()), "")
        frame = pd.read_json(path, lines=first != "[")
        return frame[frame["category"] == category].reset_index(drop=True)
    if Path(dataset).suffix in {".json", ".jsonl"}:
        raise FileNotFoundError(f"Dataset file not found: {dataset}")
    data = load_dataset(dataset, category)
    data.set_format(type="pandas")
    return data["test"][:]


def resolve_audio_file(record, manifest):
    """Resolve manifest-relative WAV paths without depending on the cwd."""
    root = Path(manifest).resolve().parent
    relative = Path(record["audio_file"])
    # Also accept original manifests that stored just a WAV basename.
    if len(relative.parts) == 1:
        relative = Path("wavs") / relative
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("audio_file must stay inside the manifest directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value
