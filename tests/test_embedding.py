"""Embedding a harness in a host: per-run overrides, ephemeral runs, attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hiveloom import construct, runner
from hiveloom.cli import app
from hiveloom.errors import ExitCode, SpecError
from hiveloom.evolve.evolver import touches_frozen
from hiveloom.models.fake import FakeModelProvider, text_response
from hiveloom.models.openai_compat import normalize_openai_response
from hiveloom.spec.loader import load_spec
from hiveloom.spec.schema import ALWAYS_FROZEN, HarnessSpec

cli = CliRunner()


class _Capturing(FakeModelProvider):
    """A fake that also records the ModelConfig each call used."""

    def complete(self, *, system, messages, tools, config):
        self.calls.append({"system": system, "config": config})
        return super().complete(system=system, messages=messages, tools=tools, config=config)


def _harness(tmp_path: Path, overridable: list[str] | None = None) -> Path:
    directory = tmp_path / "h"
    construct.init_harness(directory, name="embed-fixture", task="Return an answer.")
    if overridable is not None:
        construct.set_value(directory, "overridable", overridable)
    return directory


def _run(harness: Path, **kwargs):
    provider = kwargs.pop("provider", None) or _Capturing([text_response("ok")])
    result = runner.run_harness(
        harness, "go", provider=provider, literal_input=True, **kwargs
    )
    return result, provider


# --------------------------------------------------------------------------- #
# Spec-declared overrides
# --------------------------------------------------------------------------- #
def test_overrides_apply_in_memory_and_are_reported(tmp_path: Path):
    harness = _harness(
        tmp_path, ["system_prompt", "model.temperature", "model.max_tokens", "model.params.top_k"]
    )
    yaml_path = harness / "harness.yaml"
    original = yaml_path.read_bytes()

    baseline, _ = _run(harness, ingest=False)
    result, provider = _run(
        harness,
        ingest=False,
        overrides={
            "system_prompt": "Be terse.",
            "model.temperature": 0.2,
            "model.max_tokens": 512,
            "model.params.top_k": 5,
        },
    )

    assert yaml_path.read_bytes() == original
    call = provider.calls[0]
    assert "Be terse." in call["system"]
    assert call["config"].temperature == 0.2
    assert call["config"].max_tokens == 512
    assert call["config"].params == {"top_k": 5}
    digests = result.runtime_config["overrides"]
    assert set(digests) == {
        "system_prompt",
        "model.temperature",
        "model.max_tokens",
        "model.params.top_k",
    }
    assert "Be terse." not in json.dumps(result.runtime_config)
    assert "overrides" not in baseline.runtime_config
    # The resolved spec is what the version hash covers.
    assert result.execution.behavior_hash != baseline.execution.behavior_hash
    assert runner.run_result_payload(result)["runtime_config"]["overrides"] == digests


def test_step_instruction_override(tmp_path: Path):
    harness = _harness(tmp_path, ["loop.steps.draft.instruction"])
    construct.set_value(
        harness, "loop.steps", [{"id": "draft", "instruction": "Write a draft."}]
    )
    construct.set_value(harness, "loop.policy", "sequential_steps")
    construct.set_value(harness, "overridable", ["loop.steps.draft.instruction"])

    info = runner.dry_run(
        harness,
        "go",
        literal_input=True,
        overrides={"loop.steps.draft.instruction": "Write two lines."},
    )
    assert info["steps"][0]["instruction"] == "Write two lines."

    with pytest.raises(SpecError, match="no structured loop step"):
        construct.set_value(harness, "overridable", ["loop.steps.nope.instruction"])
        runner.dry_run(
            harness, "go", literal_input=True, overrides={"loop.steps.nope.instruction": "x"}
        )


def test_unlisted_path_and_invalid_value_are_spec_errors(tmp_path: Path):
    harness = _harness(tmp_path, ["model.temperature"])
    with pytest.raises(SpecError, match="not overridable: system_prompt"):
        _run(harness, ingest=False, overrides={"system_prompt": "x"})
    with pytest.raises(SpecError, match="temperature"):
        _run(harness, ingest=False, overrides={"model.temperature": 7})


@pytest.mark.parametrize(
    "path",
    [
        "guardrails",
        "model.provider",
        "model.id",
        "logging.redact",
        "egress.mode",
        "confinement.mode",
        "extensions",
        "hooks",
        "mcp_servers",
        "tools",
        "verify.validators",
        "evolution.auto_propose",
        "memory.max_entries",
        "overridable",
        "loop.max_turns",
    ],
)
def test_safety_paths_can_never_be_listed(tmp_path: Path, path: str):
    harness = _harness(tmp_path)
    with pytest.raises(SpecError):
        construct.set_value(harness, "overridable", [path])


def test_overridable_is_frozen_from_evolution():
    assert "overridable" in ALWAYS_FROZEN
    assert touches_frozen("overridable", set(ALWAYS_FROZEN))


def test_overridable_default_keeps_spec_shape(tmp_path: Path):
    harness = _harness(tmp_path)
    assert "overridable" not in (harness / "harness.yaml").read_text()
    assert load_spec(harness).overridable == []


def test_cli_set_and_set_file(tmp_path: Path):
    harness = _harness(tmp_path, ["system_prompt", "model.temperature"])
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("From a file.", encoding="utf-8")
    result = cli.invoke(
        app,
        [
            "run", str(harness), "--input-text", "go", "--dry-run", "--json",
            "--set", "model.temperature=0.3",
            "--set-file", f"system_prompt={prompt}",
        ],
    )
    assert result.exit_code == ExitCode.OK, result.stdout
    payload = json.loads(result.stdout)
    assert "From a file." in payload["system"]
    assert set(payload["runtime_config"]["overrides"]) == {"system_prompt", "model.temperature"}


def test_cli_set_unlisted_is_exit_3(tmp_path: Path):
    harness = _harness(tmp_path, ["model.temperature"])
    result = cli.invoke(
        app,
        ["run", str(harness), "--input-text", "go", "--json", "--set", "system_prompt=x"],
    )
    assert result.exit_code == ExitCode.SPEC_ERROR
    assert json.loads(result.stdout)["ok"] is False
    bad = cli.invoke(app, ["run", str(harness), "--input-text", "go", "--json", "--set", "oops"])
    assert bad.exit_code == ExitCode.SPEC_ERROR


# --------------------------------------------------------------------------- #
# Ephemeral runs
# --------------------------------------------------------------------------- #
def _tree(root: Path) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        snapshot[str(path.relative_to(root))] = path.read_bytes() if path.is_file() else None
    return snapshot


def test_ephemeral_run_leaves_nothing_behind(tmp_path: Path, monkeypatch):
    # Built under the suite's home, so it is unknown to the empty home below.
    harness = _harness(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HIVELOOM_HOME", str(home))
    monkeypatch.setenv("HIVELOOM_DB", str(home / "hive.db"))
    monkeypatch.delenv("HIVELOOM_TRUST")
    monkeypatch.chdir(cwd)
    # Not trusted here (no HIVELOOM_TRUST policy); this run's approval is the only one.
    scratch_root = Path(os.environ.get("TMPDIR", "/tmp"))
    before = {name: _tree(root) for name, root in
              {"home": home, "harness": harness, "cwd": cwd}.items()}
    leftovers = set(scratch_root.glob("hiveloom-ephemeral-*"))

    result, _ = _run(harness, ephemeral=True, approve_trust=lambda _path: True)

    after = {name: _tree(root) for name, root in
             {"home": home, "harness": harness, "cwd": cwd}.items()}
    assert after == before
    assert set(scratch_root.glob("hiveloom-ephemeral-*")) == leftovers
    assert not (home / "trust.json").exists()
    assert not (home / "hive.db").exists()
    assert result.status == "success"

    serialized = json.dumps(runner.run_result_payload(result))
    assert str(home) not in serialized
    assert str(tmp_path) not in serialized
    assert "hiveloom-ephemeral" not in serialized
    assert result.trace_path == "" and result.execution.trace_path == ""


def test_ephemeral_does_not_record_env_trust(tmp_path: Path, monkeypatch):
    # HIVELOOM_TRUST=always (the suite default) normally records trust on first use.
    harness = _harness(tmp_path)
    trust_file = Path(os.environ["HIVELOOM_HOME"]) / "trust.json"
    trust_file.unlink(missing_ok=True)
    _run(harness, ephemeral=True)
    assert not trust_file.exists()


def test_ephemeral_removes_scratch_when_the_run_raises(tmp_path: Path):
    harness = _harness(tmp_path)
    scratch_root = Path(os.environ.get("TMPDIR", "/tmp"))
    before = set(scratch_root.glob("hiveloom-ephemeral-*"))
    boom = FakeModelProvider([RuntimeError("boom")])
    try:
        runner.run_harness(harness, "go", provider=boom, literal_input=True, ephemeral=True)
    except Exception:  # noqa: BLE001 - the loop may report it as an error result instead
        pass
    assert set(scratch_root.glob("hiveloom-ephemeral-*")) == before


def test_ephemeral_refuses_what_must_outlive_the_run(tmp_path: Path):
    harness = _harness(tmp_path)
    for extra in ({"trace_dir": tmp_path / "t"}, {"hive_path": tmp_path / "h.db"},
                  {"resume_messages": [{"role": "user", "content": "x"}]}):
        with pytest.raises(ValueError, match="ephemeral"):
            runner.run_harness(
                harness,
                None if "resume_messages" in extra else "go",
                provider=FakeModelProvider([]),
                literal_input=True,
                ephemeral=True,
                **extra,
            )
    refused = cli.invoke(
        app, ["run", str(harness), "--input-text", "go", "--json", "--ephemeral", "--resume"]
    )
    assert refused.exit_code == ExitCode.SPEC_ERROR


def test_cli_ephemeral_with_set(tmp_path: Path):
    harness = _harness(tmp_path, ["system_prompt"])
    result = cli.invoke(
        app,
        ["run", str(harness), "--input-text", "go", "--dry-run", "--json", "--ephemeral",
         "--set", "system_prompt=Hi there"],
    )
    assert result.exit_code == ExitCode.OK, result.stdout


# --------------------------------------------------------------------------- #
# Served-model attribution
# --------------------------------------------------------------------------- #
def _served(text: str, model: str):
    return text_response(text).model_copy(update={"model": model})


def test_model_substitution_is_flagged_on_calls_and_envelope(tmp_path: Path):
    harness = _harness(tmp_path)
    requested = load_spec(harness).model.id

    same, _ = _run(harness, ingest=False, provider=FakeModelProvider([_served("a", requested)]))
    assert same.execution.model_substituted is False
    assert same.provider_calls[0]["model_substituted"] is False

    swapped, _ = _run(harness, ingest=False, provider=FakeModelProvider([_served("a", "fallback")]))
    assert swapped.execution.model_substituted is True
    assert swapped.provider_calls[0]["model_substituted"] is True
    assert runner.run_result_payload(swapped)["execution"]["model_substituted"] is True

    unreported, _ = _run(harness, ingest=False, provider=FakeModelProvider([text_response("a")]))
    assert unreported.execution.model_substituted is False


def test_response_metadata_fields_are_kept_bounded():
    data = {
        "model": "m",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "x_fallback": True,
        "x_engine": "e" * 5000,
        "x_nested": {"a": 1},
        "x_unlisted": "no",
    }
    plain = normalize_openai_response(data)
    assert "x_fallback" not in plain.provider_metadata

    kept = normalize_openai_response(
        data, metadata_fields=["x_fallback", "x_engine", "x_nested", "x_missing"]
    ).provider_metadata
    assert kept["x_fallback"] is True
    assert len(kept["x_engine"]) == 1024
    assert "x_nested" not in kept and "x_missing" not in kept and "x_unlisted" not in kept


def test_models_yaml_declares_response_metadata_fields(tmp_path: Path, monkeypatch):
    from hiveloom import ext
    from hiveloom.models.openai_compat import OpenAICompatProvider

    home = Path(os.environ["HIVELOOM_HOME"])
    (home / "models.yaml").write_text(
        "providers:\n"
        "  gateway:\n"
        "    base_url: http://localhost:9/v1\n"
        "    response_metadata_fields: [x_fallback]\n"
        "    models:\n"
        "      - {id: m1, input_cost_per_mtok: 0, output_cost_per_mtok: 0}\n",
        encoding="utf-8",
    )
    ext.reset()
    provider = ext.build_provider("gateway", tmp_path)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider._metadata_fields == ("x_fallback",)


def test_spec_class_exposes_overridable_field():
    assert "overridable" in HarnessSpec.model_fields
