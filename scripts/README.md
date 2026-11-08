# Intel image generator CLI

Bundled Python implementation. See `../SKILL.md` for concise invocation, `../README.md` for installation and runtime details, and `../AGENTS.md` for development.

```sh
uv sync --project ./scripts --locked
uv run --project ./scripts --locked intel-image-generator -p "funny elephant" --size 16x16 -o "$PWD/elephant.png"
uv run --project ./scripts --locked intel-image-generator -i prompts.txt -o "$PWD/image.png"
uv run --project ./scripts --locked pytest
```

Custom models use `-m NAMESPACE/REPOSITORY`; only compatible Diffusers Stable Diffusion text-to-image repositories supported by Optimum Intel are in scope. Outputs and model caches are outside this package; first use downloads and exports model data. GPU is default, CPU requires explicit selection.
