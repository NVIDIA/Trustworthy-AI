"""Prevent fabricated labels and mixed-run re-extraction results."""

import json

import pytest

from bbq_eval.openai.reextract_responses_langchain import (
    load_checkpoint_files,
    parse_extraction_response,
)


def row(example=1, category="Age", answer=0):
    return {
        "category": category,
        "example_id": example,
        "question_index": 1,
        "context_condition": "ambig",
        "extracted_answer": answer,
    }


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("cannot determine", -1),
        ("Answer: b", -1),
        ("banana", -1),
        ("", -1),
        ("-1", -1),
        ("(a)", 0),
        ("B) explanation", 1),
        ("c\ntext", 2),
    ],
)
def test_reextraction_requires_a_standalone_option(reply, expected):
    assert parse_extraction_response(reply) == expected


@pytest.mark.parametrize(
    "files",
    [
        ("checkpoint_optimized_run-a.jsonl", "checkpoint_optimized_run-b.jsonl"),
        ("checkpoint_worker_0_run-a.json", "checkpoint_worker_1_run-b.json"),
        ("checkpoint_worker_0_run-a.json", "checkpoint_optimized_run-a.jsonl"),
        ("checkpoint_worker_unknown.json", "checkpoint_worker_1_run-a.json"),
    ],
)
def test_directory_rejects_multiple_or_unrecognized_runs(tmp_path, files):
    for name in files:
        content = [row()]
        path = tmp_path / name
        path.write_text(json.dumps(content[0] if path.suffix == ".jsonl" else content))
    with pytest.raises(ValueError, match="specific checkpoint file"):
        load_checkpoint_files(str(tmp_path))


def test_legacy_workers_from_one_run_are_supported(tmp_path):
    for worker in (0, 1):
        path = tmp_path / f"checkpoint_worker_{worker}_abc_config_Age_dataset.json"
        path.write_text(json.dumps([row(example=worker)]))
    assert {r["example_id"] for r in load_checkpoint_files(str(tmp_path))} == {0, 1}


def test_jsonl_retries_are_deduplicated_to_latest_attempt(tmp_path):
    path = tmp_path / "checkpoint_optimized_run-a.jsonl"
    attempts = [row(answer=-1), row(category="Religion"), row(answer=2)]
    path.write_text("".join(json.dumps(attempt) + "\n" for attempt in attempts))
    for source in (path, tmp_path):
        results = load_checkpoint_files(str(source))
        assert len(results) == 2
        assert {r["category"]: r["extracted_answer"] for r in results} == {"Age": 2, "Religion": 0}


def test_explicit_file_selects_a_run_from_a_mixed_directory(tmp_path):
    first = tmp_path / "checkpoint_optimized_run-a.jsonl"
    first.write_text(json.dumps(row(answer=1)) + "\n")
    (tmp_path / "checkpoint_optimized_run-b.jsonl").write_text(json.dumps(row(answer=2)) + "\n")
    assert [r["extracted_answer"] for r in load_checkpoint_files(str(first))] == [1]
