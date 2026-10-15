"""Regression checks for generation, checkpoints, package paths and metrics."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bbq_eval.evaluate import extract_answers_predictions
from bbq_eval.evaluate import main as evaluate
from bbq_eval.utils import PACKAGE_ROOT, load_category, resolve_api_key, resolve_audio_file

pipeline = importlib.import_module("bbq_eval.openai.generate_responses_optimized")


def checkpoint(tmp_path, resume=False):
    return pipeline.CheckpointWriter(
        tmp_path, "config.yaml", ["Age", "Religion"], "data", "test", resume
    )


def record(category="Age", example=1):
    return {
        "category": category,
        "example_id": example,
        "question_index": 1,
        "context_condition": "ambig",
        "context": "Context",
        "question": "Question?",
        "ans0": "one",
        "ans1": "two",
        "ans2": "unknown",
        "extracted_answer": 0,
    }


def test_checkpoint_resume_is_explicit(tmp_path):
    original = checkpoint(tmp_path)
    original.append([record()])
    assert checkpoint(tmp_path, resume=True).is_processed(record())
    fresh = checkpoint(tmp_path)
    assert not fresh.is_processed(record())
    assert fresh.read_all() == []


def test_checkpoint_key_distinguishes_categories_and_audio(tmp_path):
    saved = checkpoint(tmp_path)
    saved.append([record()])
    assert not saved.is_processed(record("Religion"))
    assert not saved.is_processed({**record(), "audio_file": "wavs/another.wav"})


@pytest.mark.parametrize("tail", [b'{"incomplete"', b"\xe2\x82"])
def test_resume_recovers_partial_tail(tmp_path, tail):
    saved = checkpoint(tmp_path)
    saved.append([record()])
    with saved.checkpoint_path.open("ab") as stream:
        stream.write(tail)
    resumed = checkpoint(tmp_path, resume=True)
    resumed.append([record("Religion")])
    assert len(resumed.read_all()) == 2


def test_resume_rejects_malformed_complete_line(tmp_path):
    saved = checkpoint(tmp_path)
    saved.checkpoint_path.write_text("invalid\n")
    with pytest.raises(ValueError, match="Malformed"):
        checkpoint(tmp_path, resume=True)


def test_resume_adds_missing_newline(tmp_path):
    saved = checkpoint(tmp_path)
    saved.checkpoint_path.write_text(json.dumps(record()))
    resumed = checkpoint(tmp_path, resume=True)
    resumed.append([record("Religion")])
    assert len(resumed.read_all()) == 2


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("(a)", 0),
        ("B) answer", 1),
        ("c\ntext", 2),
        ("Answer: c", -1),
        ("cannot tell", -1),
        ("-1", -1),
        ("", -1),
    ],
)
def test_extraction_does_not_accept_arbitrary_first_letters(reply, expected):
    assert pipeline.parse_extraction_response(reply) == expected


def test_local_data_and_audio_resolve_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_category("Age", "dataset_small.json")) > 0
    assert len(load_category("Age", "dataset_small_ambig_95ci_10er.json")) > 0
    manifest = PACKAGE_ROOT / "data/audio-bbq-dataset/small_ambig_clean_audio.json"
    row = json.loads(manifest.read_text().splitlines()[0])
    assert resolve_audio_file(row, manifest).is_file()
    row["audio_file"] = Path(row["audio_file"]).name
    assert resolve_audio_file(row, manifest).is_file()
    row["audio_file"] = "../../outside.wav"
    with pytest.raises(ValueError):
        resolve_audio_file(row, manifest)


def test_empty_yaml_keys_fall_back_to_environment(monkeypatch):
    for name in ("API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "PARSER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert resolve_api_key({"api_key": ""}) == "test"
    assert resolve_api_key({"parser_api_key": ""}, parser=True) == "test"
    monkeypatch.setenv("PARSER_API_KEY", "parser-test")
    assert resolve_api_key({}, parser=True) == "parser-test"


@pytest.fixture
def mock_api(monkeypatch):
    state = {"main": 0, "parser": 0, "max_main": 0, "max_parser": 0, "closed": 0}

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, *, messages, **kwargs):
            stage = "parser" if messages[0]["content"].startswith("Extract which") else "main"
            state[stage] += 1
            state["max_" + stage] = max(state["max_" + stage], state[stage])
            await asyncio.sleep(0.005)
            state[stage] -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="(b)"))]
            )

        async def close(self):
            state["closed"] += 1

    monkeypatch.setattr(pipeline, "AsyncOpenAI", Client)
    monkeypatch.setenv("API_KEY", "test")
    return state


@pytest.mark.asyncio
async def test_pipeline_runs_workers_and_drains_all_results(tmp_path, mock_api):
    saved = checkpoint(tmp_path)
    frame = pd.DataFrame([record(example=i) for i in range(12)])
    frame["nested"] = [np.array([1, 2])] * 12
    await asyncio.wait_for(
        pipeline.run_pipeline(
            frame,
            {"model_name": "test-model"},
            saved,
            main_concurrency=3,
            parser_concurrency=2,
            queue_size=2,
            window_size=2,
            checkpoint_interval=5,
        ),
        timeout=5,
    )
    rows = saved.read_all()
    assert len(rows) == 12
    assert all(r["extracted_answer"] == 1 and r["nested"] == [1, 2] for r in rows)
    assert mock_api["max_main"] == 3
    assert mock_api["max_parser"] == 2
    assert mock_api["closed"] == 2


@pytest.mark.asyncio
async def test_writer_failure_cancels_pipeline(tmp_path, mock_api, monkeypatch):
    saved = checkpoint(tmp_path)

    def fail(records):
        raise OSError("disk full")

    monkeypatch.setattr(saved, "append", fail)
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(
            pipeline.run_pipeline(
                pd.DataFrame([record(example=i) for i in range(20)]),
                {"model_name": "test-model"},
                saved,
                main_concurrency=2,
                parser_concurrency=2,
                queue_size=1,
                checkpoint_interval=1,
            ),
            timeout=5,
        )
    assert mock_api["closed"] == 2


@pytest.mark.asyncio
async def test_invalid_concurrency_fails_before_requests(tmp_path):
    with pytest.raises(ValueError, match="main_concurrency"):
        await pipeline.run_pipeline(pd.DataFrame(), {}, checkpoint(tmp_path), main_concurrency=0)


@pytest.fixture
def local_run(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  model_name: test-model\n")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([record(example=1), record(example=2)]))
    return {
        "categories": ["Age"],
        "config_file": str(config),
        "output_dir": str(tmp_path / "results"),
        "dataset": str(dataset),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"main_concurrency": 0},
        {"parser_concurrency": -1},
        {"window_size": 0},
        {"checkpoint_interval": 0},
        {"main_concurrency": True},
    ],
)
async def test_rejected_options_preserve_existing_run(local_run, mock_api, monkeypatch, invalid):
    output = Path(await pipeline.main_async(**local_run))
    saved = next((output.parent / "checkpoints").glob("*.jsonl"))
    original_checkpoint, original_output = saved.read_bytes(), output.read_bytes()

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid options must fail before loading the dataset")

    monkeypatch.setattr(pipeline, "load_all_categories", unexpected_load)
    with pytest.raises(ValueError, match="positive integer"):
        await pipeline.main_async(**local_run, **invalid)
    assert saved.read_bytes() == original_checkpoint
    assert output.read_bytes() == original_output


@pytest.mark.asyncio
async def test_missing_credentials_preserve_existing_run(local_run, mock_api, monkeypatch):
    output = Path(await pipeline.main_async(**local_run))
    saved = next((output.parent / "checkpoints").glob("*.jsonl"))
    original = saved.read_bytes()
    for name in ("API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "PARSER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Set API_KEY"):
        await pipeline.main_async(**local_run)
    assert saved.read_bytes() == original


@pytest.mark.asyncio
async def test_client_setup_failure_preserves_existing_run(local_run, mock_api, monkeypatch):
    output = Path(await pipeline.main_async(**local_run))
    saved = next((output.parent / "checkpoints").glob("*.jsonl"))
    original = saved.read_bytes()

    def invalid_client(**kwargs):
        raise ValueError("Invalid client configuration")

    monkeypatch.setattr(pipeline, "AsyncOpenAI", invalid_client)
    with pytest.raises(ValueError, match="Invalid client"):
        await pipeline.main_async(**local_run)
    assert saved.read_bytes() == original


@pytest.mark.asyncio
async def test_resume_retries_only_failures_and_outputs_one_record_per_example(
    local_run, mock_api, monkeypatch
):
    attempts = {}
    original = pipeline.MainModelProducer.process_one

    async def fail_once(producer, row):
        example = row["example_id"]
        attempts[example] = attempts.get(example, 0) + 1
        if example == 1 and attempts[example] == 1:
            return {"row": row, "model_response": "ERROR: model request failed", "error": True}
        return await original(producer, row)

    monkeypatch.setattr(pipeline.MainModelProducer, "process_one", fail_once)
    output = Path(await pipeline.main_async(**local_run))
    assert sorted(r["extracted_answer"] for r in json.loads(output.read_text())) == [-1, 1]
    await pipeline.main_async(**local_run, resume=True)
    results = json.loads(output.read_text())
    assert len(results) == 2
    assert {r["example_id"] for r in results} == {1, 2}
    assert all(r["extracted_answer"] == 1 for r in results)
    assert attempts == {1: 2, 2: 1}
    saved = next((output.parent / "checkpoints").glob("*.jsonl"))
    assert len(saved.read_text().splitlines()) == 3  # Preserve failed-attempt history.
    await pipeline.main_async(**local_run, resume=True)
    assert attempts == {1: 2, 2: 1}


@pytest.mark.parametrize("answer", [-1, None])
def test_checkpoint_retries_unresolved_parser_answers(tmp_path, answer):
    saved = checkpoint(tmp_path)
    saved.append([{**record(), "extracted_answer": answer}])
    resumed = checkpoint(tmp_path, resume=True)
    assert not resumed.is_processed(record())
    resumed.append([{**record(), "extracted_answer": 2}])
    assert resumed.is_processed(record())
    reloaded = checkpoint(tmp_path, resume=True)
    assert reloaded.is_processed(record())
    assert [r["extracted_answer"] for r in reloaded.read_all()] == [2]


def test_invalid_extractions_do_not_become_option_c():
    frame = pd.DataFrame(
        {
            "extracted_answer": [0, 1, 2, -1, None, "bad"],
            "label": [0] * 6,
            "ans0_info": ["one"] * 6,
            "ans1_info": ["two"] * 6,
            "ans2_info": ["three"] * 6,
        }
    )
    result = extract_answers_predictions(frame, "model_response")
    assert result["pred_cat"].tolist() == ["one", "two", "three"]
    assert result["acc"].tolist() == [1, 0, 0]


def test_metrics_accept_nested_json_and_json_lines(tmp_path):
    frame = load_category("Age", "dataset_small_ambig_95ci_10er.json").head(6)
    frame["extracted_answer"] = frame["label"]
    metadata = PACKAGE_ROOT / "additional_data/additional_metadata.csv"
    for lines in (False, True):
        source = tmp_path / f"responses-{lines}.json"
        frame.to_json(source, orient="records", lines=lines)
        out = tmp_path / str(lines)
        evaluate(metadata, source, out, "model_response", use_categories=True)
        metrics = pd.read_csv(out / "bbq_metrics.csv")
        assert metrics["category"].tolist() == ["Age"]
        assert metrics["acc_bias"].tolist() == [0.0]
        assert pd.read_csv(out / "bbq_metrics_overall.csv")["acc_bias"].tolist() == [0.0]
