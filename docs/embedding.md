# Embedding hiveloom in a host process

A host service that runs one harness per request (behind its own gateway,
inside a throwaway sandbox) uses the same library as the CLI. Nothing here is
new machinery; it is the recommended way to combine what exists, so a host does
not copy and rewrite the harness folder or scrub results.

## One run, one fresh interpreter

Run each harness in a short-lived child process, not in the host's own
interpreter. The harness folder carries code (tools, validators, hooks) that is
imported into the process that runs it; a child keeps that code, its imports and
its environment away from the host and from the next request.

- **Fresh interpreter.** `python -I -m your_worker` (isolated mode: no
  user site, no `PYTHON*` variables, no current-directory imports).
- **Environment allowlist.** Build the child's `env` from scratch rather than
  inheriting: `PATH`, the one provider key the harness needs
  (`api_key_env` of its provider), and `HIVELOOM_HOME` pointing at a directory
  that holds only the provider declarations (`models.yaml`) the run needs.
- **Hard backstop.** Start the child in its own process group
  (`start_new_session=True`) and kill the group (`os.killpg`) when the
  deadline passes. Stopping a run gracefully is a request, not a guarantee (below).

## `ephemeral=True`

```python
result = run_harness(
    harness_dir,
    input_value=task,
    literal_input=True,
    ephemeral=True,
    overrides={"system_prompt": prompt, "model.temperature": 0.2},
    control=control,
    approve_trust=lambda _path: True,   # holds for this run only
)
```

An ephemeral run persists nothing (`hiveloom run --ephemeral` is the CLI form):

- the journal, spill store, notes and a scratch Hive live in a temporary
  directory that is removed before `run_harness` returns, including when the
  run raises;
- no Hive ingest, auto-propose or reflection (they read the Hive);
- a trust approval, or `HIVELOOM_TRUST=always`, applies to this run and never
  writes `trust.json`;
- `trace_path` is empty on the result and on `execution`, so
  `run_result_payload(result)` contains no local path of the run;
- a delegated peer run inherits `ephemeral`.

Consequences: `recall_runs` and directory-mode delegation start from an empty
Hive, `propose_memory` writes to the scratch Hive and is discarded, and
`resume_messages`, `trace_dir` and `hive_path` are refused (each names state
that must outlive the run). `on_event` still sees every trace event, so a host
that wants the journal can collect it there. Tools the harness itself declares
(`file_write`, a code tool, `shell`) act on whatever they are pointed at;
ephemeral does not sandbox them. `confinement` does.

## Spec-declared overrides

Do not copy the folder and rewrite `harness.yaml` per run. Declare what may vary
in the spec, once, through the CLI:

```bash
hiveloom set overridable '["system_prompt", "model.temperature", "model.max_tokens"]'
```

and pass values per run: `run_harness(..., overrides={...})` or
`hiveloom run --set model.temperature=0.2 --set-file system_prompt=prompt.txt`.
Only the closed allowlist can be listed (see `docs/spec.md`); the safety layer
cannot. The override applies to an in-memory spec that is fully re-validated, a
path not listed or a rejected value is a spec error (exit 3), `harness.yaml` is
never written, and the result's `runtime_config.overrides` carries each applied
path with a hash of its value. Overrides are part of the resolved spec, so the
version hash separates runs with different overrides. Use `--model` /
`--provider` (`model_override` / `provider_override`) to change the executor.

## Stopping and deadlines

`RunControl.request_stop(reason)` from a heartbeat thread (for example when the
parent stops answering or a request is cancelled) ends the run at the next turn
boundary with `status: "stopped"`. An in-flight model call is not interrupted,
so the stop can lag by one call: keep the process-group kill as the backstop.
Put a ceiling on the run itself with the `max_wall_clock_seconds` guardrail.

## Shipping the harness

`hiveloom package ./h` writes one `<name>-<hash>.zip`. It contains the harness
folder under `<name>/` — `harness.yaml`, code assets, skills, `hiveloom.lock`
(runtime version, harness hash, packs) and, with `--docker`, a `Dockerfile` and
`.dockerignore` — and leaves out `.hiveloom/` (trust store, traces), `.env*`
files other than `.env.example`/`.env.template`, `.venv`, `.git`,
`__pycache__`, `*.pyc` and `*.zip`. The receiving machine unzips it and
trusts it (`approve_trust`, `hiveloom trust`, or `HIVELOOM_TRUST=always`)
before any of its code loads.

## Which model answered

Every provider call record has `requested_model`, `effective_model` (what the
provider reported) and `model_substituted`; the run's `execution.model_substituted`
is true when any call differed. The comparison is an exact string match, so a
provider that reports a dated snapshot of the alias you asked for reads as
substituted.

A gateway that flags a fallback with its own response field can be asked to
keep it. In `models.yaml`:

```yaml
providers:
  gateway:
    base_url: https://gateway.internal/v1
    response_metadata_fields: [x_served_by]   # up to 8 top-level fields
```

(`OpenAICompatProvider(response_metadata_fields=[...])` is the same option in
code). Scalar values are kept, strings clamped, in `provider_metadata` on the
`model_response` event, which reaches a host through `on_event`; it is not on
the result.
