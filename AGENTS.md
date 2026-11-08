# Development guide

## Scope and layout

This repository is an Agent Skill for local text-to-image generation. Keep the skill portable across Pi, Claude Code, and Codex. Do not add host-specific tool permissions to SKILL.md.

```text
SKILL.md                     Runtime instructions and skill metadata
README.md                    Purpose, installation, usage, verification
AGENTS.md                    Development guidance
scripts/
  pyproject.toml             Python package, CLI entry point, uv indexes
  uv.lock                    Locked dependencies
  .python-version            Selected Python version
  README.md                  Bundled Python package notes
  src/intel_image_generator/
    cli.py                   Argument validation, resize/encode, atomic save
    generation.py            Device check, model export/cache, inference
  tests/
    test_cli.py              Fake-backend boundary tests
    test_generation.py       Mocked model/cache tests
```

## Commands

Run from the repository root:

```sh
uv sync --project ./scripts --locked
uv run --project ./scripts --locked pytest
uv run --project ./scripts --locked intel-image-generator --help
```

If another virtual environment is active, prefix these commands with `env -u VIRTUAL_ENV` on Linux. Never use `--active` or modify the agent harness environment. The project environment belongs at `scripts/.venv/`; do not move an existing virtual environment between directories. Recreate it with uv.

Manage dependencies inside `scripts/`, for example `uv add --project ./scripts PACKAGE`. Update and review both pyproject.toml and uv.lock. PyTorch and torchvision use the explicit CPU wheel index; diffusion inference uses OpenVINO. Do not introduce CUDA packages.

## Behavioral contracts

- Generation requires exactly one of `-p/--prompt` and `-i/--input` plus `-o/--output`; `--info` is long-only and model-free. Prompt files are processed sequentially with one image per prompt. Never write image bytes to stdout; keep diagnostics on stderr.
- Validate prompt, numeric bounds, and destination before loading heavy libraries.
- Encode a complete RGB PNG in memory. Write to a temporary file beside the destination, then use atomic replacement. Preserve existing output on failure; clean up temporary files.
- Generate at 512x512; treat `--size` as final dimensions with centered crop and LANCZOS resize.
- Auto-select NPU → integrated Intel GPU → CPU, or discrete Intel GPU → CPU. Use CPU when GPU architecture detection fails; explicit device choices remain strict. Never retry compilation/inference on a different device silently.
- Keep `--info` a model-free preflight: stdout contains device/model JSON, diagnostics stay on stderr, and no prompt/output, Hub request, weights, or compilation is needed. Selection does not establish model compatibility.
- Retain model safety filtering. Do not remove filters or modify graphs to bypass compatibility problems.
- Keep exported models outside the repository. Use version-keyed caches, a lock, required-file validation, a completion marker, and atomic export publication. Do not reuse incomplete exports.
- Keep cached/local loads offline-capable. Do not hide unexpected exceptions or suppress all upstream logging.
- Resolve script paths against the skill location, but outputs against the user's workspace. Document absolute output paths in agent examples.

## Tests and real inference

Unit tests must not require network access, model files, or a GPU. Mock the backend and heavy modules. Cover parsing/bounds, model ID rules and aliases/conflicts, model-free info output and failures, help without runtime imports, option propagation, exact PNG dimensions/RGB, stdout isolation, overwrite/write failures, cache separation/reuse/incomplete exports, original versus pre-exported models, offline cache reuse, unsupported pipeline configuration, export cleanup, and device checks.

Run the full suite after changes. For path/layout changes, also run help from an unrelated working directory using an absolute `--project` path.

A real inference smoke test downloads several GB on first use. Run it only with approval for model downloads and compute use. Do not collect additional models. Use a temporary absolute output path and a fixed seed:

```sh
uv run --project ./scripts --locked intel-image-generator \
  --size 16x16 --seed 42 --steps 4 -o /tmp/intel-image-generator-smoke.png -p "funny elephant"
```

Decode the PNG and verify dimensions, RGB, empty stdout, and exit status. Repeat with `HF_HUB_OFFLINE=1` to check cache reuse. Record cold versus cached context; do not label one observation a benchmark. CPU testing requires explicit opt-in.

## Workflow and safety

Make scoped changes, inspect diffs, and report tests and blockers. Use subagents only when Alex explicitly requests delegation. When authorized, use `worker` for code and `browser-research` for web fact collection; keep coordination and acceptance in the main agent.

Ask before sudo, system driver changes, or repository additions. Do not force-install incompatible packages, silently change model/backend, commit automatically, or install this skill into agents' config directories without approval.

Do not commit `.venv`, Python/test caches, generated PNGs, downloaded weights, or export artifacts. Do not include environments/caches in skill distributions. Keep installation docs and SKILL.md in sync when paths or CLI behavior change.
