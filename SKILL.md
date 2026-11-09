---
name: intel-image-generator
description: Generate images, icons, and visual assets locally on Intel hardware.
---

# Intel image generator

Resolve `scripts/` relative to this file. Use an absolute output path in the user's workspace.

## Check selection first

Before every generation, run `--info` with the same model/device options intended for generation:

```sh
SKILL_DIR="/absolute/path/to/intel-image-generator"
env -u VIRTUAL_ENV uv run --project "$SKILL_DIR/scripts" --locked intel-image-generator --info
```

This prints JSON containing `device` and `model`. It does not download, validate, compile, or run the model. Check the selected device and model against the user's request and the compatibility table below. Do not claim an untested model works because `--info` succeeds. Use the reported model unless the user requests an override. Do not substitute a different model silently.

Then generate with the same options, prompt, and output. Omit `--device` to keep automatic selection, or use an explicit device when the user requests it:

```sh
env -u VIRTUAL_ENV uv run --project "$SKILL_DIR/scripts" --locked intel-image-generator \
  --size 512x512 --seed 42 --steps 20 \
  -p "A friendly elephant, flat illustration" \
  -o /absolute/path/to/output.png
```

Check the exit status and that the output exists; then return its path. If changing model/device options, rerun `--info` first.

## Options

`--info` (long-only) needs no prompt/output. Generation requires exactly one of `-p/--prompt TEXT` or `-i/--input PATH` (UTF-8, one prompt per nonblank line), plus `-o/--output PATH`. Prompts from a file are generated sequentially, one image per prompt, saving multiple images as stem-1.ext, stem-2.ext, etc. Each file is atomic; all destinations are prevalidated and a failed run may retain earlier completed images. `-m/--model NAMESPACE/REPOSITORY` selects a compatible model; `--model-dir PATH` selects a local export (mutually exclusive). Other options: `--size WIDTHxHEIGHT`, `--seed INTEGER`, `--steps INTEGER`, `--device NPU|GPU|CPU`.

For reference-image conditioning, use `-r/--reference PATH` with `--model stable-diffusion-v1-5/stable-diffusion-v1-5`. This uses IP-Adapter for text-to-image, not img2img; it does not guarantee character identity or exact style. It cannot be combined with `--source`, `--strength`, or `--model-dir`. The required model must be explicit even when it is the default, including for NPU where the default is different. First use downloads several GB and creates separate exports. Do not silently change the model.

For img2img, use `-s/--source PATH` for the starting image and optionally `--strength FLOAT` (greater than 0 and at most 1, default 0.75). `--strength` requires `--source`. `--style` was removed; `-r` now selects reference conditioning and `-s` selects an img2img source.

Without `--device`, integrated Intel GPU systems prefer NPU → GPU → CPU; discrete Intel GPU systems prefer GPU → NPU → CPU. Unknown GPU type or failed GPU-property detection falls back to CPU. Explicit choices are strict. Compilation/inference errors do not trigger device retries.

## NPU model compatibility

This is the single model-compatibility list for this skill. Evidence applies to the exact model and software setup, not every derivative.

| Hugging Face model | NPU status for this CLI |
|---|---|
| [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | **CPU/GPU default; NPU-tested:** exported locally; successful generation on Lunar Lake Intel AI Boost with OpenVINO 2026.4.1 / Optimum Intel 1.27.0. |
| [`OpenVINO/stable-diffusion-v1-5-fp16-ov`](https://huggingface.co/OpenVINO/stable-diffusion-v1-5-fp16-ov) | Compatible SD pipeline candidate; **not NPU-tested here**. Do not promise successful compilation. |
| [`OpenVINO/stable-diffusion-v1-5-int8-ov`](https://huggingface.co/OpenVINO/stable-diffusion-v1-5-int8-ov) | **NPU default; successfully generated one image on Lunar Lake with OpenVINO 2026.4.1 / Optimum Intel 1.27.0.** |
| [`Intel/sd-1.5-square-quantized`](https://huggingface.co/Intel/sd-1.5-square-quantized) | Intel's card names NPU, but its repository lacks the required Diffusers `model_index.json`/component layout. **Not directly usable with this CLI's `--model`.** |

SDXL and FLUX require other pipeline support and are not selectable NPU alternatives in this CLI. Do not download candidates automatically merely to test compatibility.

The model safety checker is disabled. See README.md for setup, first-run costs, model/license details, and troubleshooting. Do not silently fall back to CPU or make unapproved system/driver changes.
