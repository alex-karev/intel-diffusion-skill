# Intel image generator skill

Generate PNG images locally from text prompts using Stable Diffusion 1.5 and OpenVINO. This Agent Skill bundles instructions and a uv-managed Python CLI. It detects Intel accelerators with CPU as the safe fallback, supports exact output dimensions and seeds, and saves files atomically with `-o/--output`.

No server, web UI, cloud inference, or CUDA dependency. Use `--device CPU` to force CPU generation. SD 1.5 is a functional baseline, not a guarantee of polished logos or legible text.

## Requirements

- Linux, `uv`, and Python 3.12 or newer. The project selects Python 3.12.
- Working Intel NPU/GPU runtime to use those accelerators; CPU is the fallback when no supported accelerator is detected.
- Network access on first use, plus disk space for dependencies and model files. Observed model storage was about 5.2 GiB in the Hugging Face cache and 7.2 GiB for the OpenVINO export.

Installing the skill does not install system GPU drivers. Review the code before enabling it. Ask your system administrator before changing driver packages.

## Install the skill

Keep this entire directory, including `scripts/`. Copying only SKILL.md will not work. No `uv tool install` is needed: the agent runs the bundled project directly.

The examples below install from a local checkout. Run them from this repository's root. Pick the location for your agent; do not install duplicate copies into locations the same agent scans. When upgrading, replace/rename an existing `openvino-image` skill directory; do not keep both skill names active.

### Pi agent

```sh
mkdir -p ~/.pi/agent/skills/intel-image-generator
tar --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=build --exclude=dist --exclude='*.egg-info' \
  -cf - SKILL.md README.md AGENTS.md scripts |
  tar -xf - -C ~/.pi/agent/skills/intel-image-generator/
```

Start Pi again or run `/reload`. Invoke it explicitly:

```text
/skill:intel-image-generator Generate a 128x128 elephant icon and save it as elephant.png
```

For project-only use, copy the same files into `.pi/skills/intel-image-generator/` in the target project instead.

### Claude Code

```sh
mkdir -p ~/.claude/skills/intel-image-generator
tar --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=build --exclude=dist --exclude='*.egg-info' \
  -cf - SKILL.md README.md AGENTS.md scripts |
  tar -xf - -C ~/.claude/skills/intel-image-generator/
```

Start a new Claude Code session and invoke:

```text
/intel-image-generator Generate a 128x128 elephant icon and save it as elephant.png
```

For project-only use, use `.claude/skills/intel-image-generator/` instead.

### Codex

```sh
mkdir -p ~/.agents/skills/intel-image-generator
tar --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=build --exclude=dist --exclude='*.egg-info' \
  -cf - SKILL.md README.md AGENTS.md scripts |
  tar -xf - -C ~/.agents/skills/intel-image-generator/
```

Codex discovers skills in `~/.agents/skills/` and project `.agents/skills/`. Invoke it with `$intel-image-generator` or select it through `/skills`. Restart Codex if it does not appear.

Pi also scans `~/.agents/skills/`, so this location can serve both Pi and Codex. Do not also install a second Pi copy with the same skill name.

### Prepare the Python environment

After copying, run this with the installed skill's absolute path:

```sh
SKILL_DIR="$HOME/.pi/agent/skills/intel-image-generator" # adjust for your installation
uv sync --project "$SKILL_DIR/scripts" --locked
uv run --project "$SKILL_DIR/scripts" --locked intel-image-generator --help
```

The first generation downloads and exports the model. Later runs use the local cache. The install commands exclude generated environments and caches. Python virtual environments are not portable between paths; uv creates a fresh environment at the installed location.

## CLI usage

From this checkout, or using the installed skill's absolute `scripts` path:

Check the chosen device/model before generating:

```sh
uv run --project ./scripts --locked intel-image-generator --info
# Example: {"device": "NPU", "model": "OpenVINO/stable-diffusion-v1-5-int8-ov"}

uv run --project ./scripts --locked intel-image-generator \
  --size 16x16 --seed 42 -o elephant.png -p "funny elephant"
```

`--info` accepts the same `--device`, `--model`, or `--model-dir` selection options. It prints JSON to stdout and needs neither a prompt nor an output path; it is long-only. It detects hardware but never contacts Hugging Face, loads model weights, compiles, or generates. It reports selection, not model compatibility or successful NPU compilation. Normal uv dependency setup may still occur if the Python environment is missing.

Without `--device`, an integrated Intel GPU system prefers NPU → GPU → CPU; a discrete Intel GPU system prefers GPU → NPU → CPU. Unknown GPU type or failed GPU-property detection falls back to CPU. Explicit device choices are strict, and compilation/inference errors never trigger device retries.

`--project` selects the environment; relative output paths resolve against your current working directory, not the skill directory.

| Option | Behavior |
|---|---|
| `-p`, `--prompt TEXT` | Single prompt (mutually exclusive with input) |
| `-i`, `--input PATH` | UTF-8 lines, blank lines ignored; generated sequentially, one output per prompt |
| `--info` | Print selected device/model JSON without generation (long-only) |
| `-o`, `--output` | Required for generation; parent must exist; `-` is unsupported. Multiple prompts produce stem-1.ext, stem-2.ext, etc.; each image is atomic but the whole run is not transactional. |
| `-r`, `--reference PATH` | IP-Adapter reference image for style/visual conditioning of text-to-image. Requires explicit original SD 1.5 model; incompatible with `--source`, `--strength`, and `--model-dir`. |
| `-s`, `--source PATH` | Starting image for img2img, transformed according to the prompt; incompatible with `--reference`. |
| `--strength FLOAT` | Img2img denoising strength, greater than 0 and at most 1; requires `--source`, defaults to 0.75. |
| `--size WIDTHxHEIGHT` | Final dimensions, 1–4096 per side; default 512x512 |
| `--seed INTEGER` | Optional seed, 0–2^63-1 |
| `--steps INTEGER` | 1–1000, default 20 |
| `--device NPU`, `GPU`, or `CPU` | Explicit device override; omission auto-detects with CPU fallback |
| `-m`, `--model NAMESPACE/REPOSITORY` | Compatible Hugging Face Diffusers Stable Diffusion text-to-image repository; defaults to SD 1.5 except NPU uses the int8 OpenVINO model |
| `--model-dir PATH` | Existing local OpenVINO export; mutually exclusive with `--model` |

For reference conditioning, specify the original SD 1.5 model explicitly, including on NPU (whose default model is different):

```sh
uv run --project ./scripts --locked intel-image-generator \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --reference /absolute/path/to/reference.png \
  -p "A cat riding a skateboard" -o /absolute/path/to/output.png
```

`--reference` conditions text-to-image generation via IP-Adapter; it is not img2img and does not guarantee character identity or exact style reproduction. Use `--source /absolute/path/to/source.png --strength 0.75` instead to start from an image with img2img. These modes cannot be combined. The former `--style` option was removed; `-r` now means reference conditioning and `-s` now means img2img source.

Generation uses a 512x512 canvas, then centered crop and LANCZOS resize. Successful output atomically replaces an existing destination; failure preserves it. stdout stays empty during generation; diagnostics go to stderr.

The OpenVINO export cache lives under `$XDG_CACHE_HOME/openvino-image/` or `~/.cache/openvino-image/` for the default SD 1.5 model; custom models are isolated under the `intel-image-generator` namespace with model-specific versioned keys. The default cache key and marker preserve compatibility with exports made by the previous `openvino-image` CLI, so renaming does not force a re-export. Reference conditioning downloads the pinned SD 1.5 source, IP-Adapter weights, and image encoder, then builds separate versioned base and style exports under `intel-image-generator`. Allow several additional GB for these assets and exports. Style graphs are compiled for the selected device on each run; persistent compiler caching is not enabled for them. Hugging Face uses its standard cache. After setup, `HF_HUB_OFFLINE=1` requests offline reuse.

## Model compatibility, setup, and verification

Custom model selection supports only compatible Diffusers `StableDiffusionPipeline` text-to-image repositories handled by the installed Optimum Intel pipeline. The CLI reads the repository's `model_index.json` and rejects unsupported pipeline classes; it does not enable arbitrary remote code. Hub IDs must use `namespace/name` form. Private or gated models require existing Hugging Face authentication and accepted model terms. LoRAs, standalone checkpoint files, arbitrary architectures, and dedicated inpainting pipelines are outside scope. Model license and access terms remain the user's responsibility.

The implementation uses OpenVINO and Optimum Intel. Original Diffusers repositories are exported on first use; compatible OpenVINO-export repositories can load directly. First use downloads source model files and writes an OpenVINO export, requiring several GB and potentially significant time. Subsequent runs use validated version-keyed caches. Export writes are lock-protected and published atomically; incomplete exports are discarded. Generation always uses a 512x512 canvas, then centered crop and LANCZOS resize. NPU execution uses static shapes before compilation. Driver/runtime prerequisites are not installed by this skill. The single NPU model compatibility table lives in [SKILL.md](SKILL.md#npu-model-compatibility); it distinguishes tested models from candidates and incompatible repository layouts.

Model: [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), subject to the [CreativeML OpenRAIL-M license](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE). The verified source revision was `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The model safety checker is disabled; generated output is not filtered.

GPU smoke tests passed on Debian 13, Intel Core Ultra 7 258V / Intel Arc iGPU, OpenVINO 2026.4.1, and Optimum Intel 1.27.0. Cold setup took about 22 minutes, mostly downloading; a cached four-step offline run took about 20 seconds including loading, compilation, and generation. These are observations, not latency guarantees. Those earlier observations are not guarantees for current code or other systems.

On that host, Alex installed Intel OpenCL 26.40.40196.8, IGC 2.42.6, and libigdgmm12 22.10.2 from Intel's Ubuntu-targeted release packages. This is a tested local configuration, not an official Debian installation recipe.

Optimum Intel 1.27.0 can emit local-export detection and missing `refs` path warnings even when offline loading succeeds. An `optimum` multiple-distributions warning was also observed. Check the exit status and output; do not ignore unrelated failures.

## Development and references

See [AGENTS.md](AGENTS.md) for the project layout, tests, and development rules. Runtime instructions live in [SKILL.md](SKILL.md).

Installation locations follow the [Pi skills docs](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/skills.md), [Claude Code skills docs](https://code.claude.com/docs/en/skills), and [Codex skills docs](https://developers.openai.com/codex/skills).
