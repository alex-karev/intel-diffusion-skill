"""OpenVINO Stable Diffusion loading, export caching, and image generation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CACHE_SCHEMA = 1
SUPPORTED_PIPELINES = {"StableDiffusionPipeline", "OVStableDiffusionPipeline"}


def _say(message: str) -> None:
    print(f"intel-image-generator: {message}", file=sys.stderr)


def _cache_path(model: str = MODEL) -> Path:
    versions = {}
    for package in ("optimum-intel", "openvino", "diffusers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required package {package} is not installed") from exc
    key = hashlib.sha256(
        json.dumps([model, CACHE_SCHEMA, versions], sort_keys=True).encode()
    ).hexdigest()[:20]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    # Preserve the previous default cache path, key, and marker for existing SD 1.5 exports.
    namespace = "openvino-image" if model == MODEL else "intel-image-generator"
    prefix = "sd15" if model == MODEL else hashlib.sha256(model.encode()).hexdigest()[:16]
    return root / namespace / f"{prefix}-{key}"


def _valid_export_without_marker(path: Path) -> bool:
    required = [path / "model_index.json"]
    for component in ("unet", "text_encoder", "vae_decoder"):
        required.extend(
            (path / component / "openvino_model.xml", path / component / "openvino_model.bin")
        )
    required.extend(
        (path / "tokenizer" / "tokenizer_config.json", path / "scheduler" / "scheduler_config.json")
    )
    return all(item.is_file() for item in required)


def _valid_export(path: Path, model: str = MODEL) -> bool:
    try:
        marker = json.loads((path / ".openvino-image-complete.json").read_text())
    except (OSError, ValueError):
        return False
    return marker == {"model": model, "schema": CACHE_SCHEMA} and _valid_export_without_marker(path)


def _is_openvino_repository(files: set[str]) -> bool:
    return all(
        f"{component}/openvino_model.{extension}" in files
        for component in ("unet", "text_encoder", "vae_decoder")
        for extension in ("xml", "bin")
    )


def _model_config(model: str) -> tuple[dict[str, Any], set[str]]:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        config_path = hf_hub_download(model, "model_index.json")
        config = json.loads(Path(config_path).read_text())
        files = set(list_repo_files(model))
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot access model {model!r}; it may be private or gated. Authenticate with Hugging Face and accept its license."
        ) from exc
    except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, HfHubHTTPError, OSError) as exc:
        raise RuntimeError(
            f"Cannot read model configuration for {model!r} ({type(exc).__name__}). "
            "Check the repository ID, network, and Hugging Face authentication."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model {model!r} has an invalid model_index.json") from exc

    pipeline = config.get("_class_name")
    if pipeline not in SUPPORTED_PIPELINES:
        raise RuntimeError(
            f"Unsupported model pipeline {pipeline!r}; expected a Diffusers StableDiffusionPipeline repository"
        )
    return config, files


def _pipeline_class(img2img: bool = False):
    from optimum.intel.openvino import OVStableDiffusionPipeline
    if not img2img:
        return OVStableDiffusionPipeline
    from optimum.intel.openvino import OVStableDiffusionImg2ImgPipeline
    return OVStableDiffusionImg2ImgPipeline


def _load_cached(cache: Path, device: str, img2img: bool = False):
    if img2img and not all((cache / "vae_encoder" / f"openvino_model.{ext}").is_file() for ext in ("xml", "bin")):
        raise RuntimeError(f"Image-to-image requires an encoder-bearing OpenVINO export with vae_encoder at {cache}; use a compatible --model or --model-dir")
    return _pipeline_class(img2img).from_pretrained(
        str(cache), export=False, compile=False, device=device, local_files_only=True, safety_checker=None
    )


def _publish_preexported(model: str, cache: Path) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}-", dir=cache.parent))
    try:
        try:
            snapshot = Path(snapshot_download(model, local_files_only=False))
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            raise RuntimeError(
                f"Cannot access model {model!r}; it may be private or gated. "
                "Authenticate with Hugging Face and accept its license."
            ) from exc
        except (HfHubHTTPError, OSError) as exc:
            raise RuntimeError(
                f"Cannot download OpenVINO model {model!r} ({type(exc).__name__}). "
                "Check network access and Hugging Face authentication."
            ) from exc
        # Snapshot download is cached by HF; copy a complete repository before publishing our marker.
        shutil.copytree(snapshot, temporary, dirs_exist_ok=True)
        if not _valid_export_without_marker(temporary):
            raise RuntimeError("pre-exported model is incomplete: required OpenVINO files are missing")
        (temporary / ".openvino-image-complete.json").write_text(
            json.dumps({"model": model, "schema": CACHE_SCHEMA})
        )
        os.replace(temporary, cache)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _compiled_cache_path(model: str, device: str, shape_policy: str) -> Path:
    versions = {}
    for package in ("openvino", "optimum-intel"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    identity = json.dumps([model, device, versions, shape_policy], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "intel-image-generator" / "compiled" / key


def _configure_compiled_cache(pipe, model: str, device: str, shape_policy: str) -> None:
    """Set Optimum's per-component CACHE_DIR only when this device advertises support."""
    try:
        import openvino as ov
        core = ov.Core()
        capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
        supported = core.get_property(device, "SUPPORTED_PROPERTIES")
        if "EXPORT_IMPORT" not in capabilities or "CACHE_DIR" not in supported:
            _say(f"compiled-model caching unavailable on {device}; compiling without disk cache")
            return
    except Exception as exc:
        _say(f"cannot verify compiled-model caching support on {device} ({exc}); compiling without disk cache")
        return

    cache = _compiled_cache_path(model, device, shape_policy)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        # Optimum diffusion parts inherit this shared dictionary and pass it to Core.compile_model.
        config = getattr(pipe, "ov_config", None)
        if config is None:
            config = {}
            pipe.ov_config = config
        config["CACHE_DIR"] = str(cache)
        # OVBaseModel copies ov_config during construction, so update each compiling
        # component's effective config as well as the pipeline-level default.
        for component in getattr(pipe, "components", {}).values():
            component_config = getattr(component, "ov_config", None)
            if component_config is not None:
                component_config["CACHE_DIR"] = str(cache)
        _say(f"compiled-model cache enabled at {cache} (cache hit not verified)")
    except OSError as exc:
        _say(f"cannot prepare compiled-model cache at {cache} ({exc}); compiling without disk cache")


def _default_pipeline(device: str, model: str, img2img: bool = False):
    from filelock import FileLock
    from optimum.intel.openvino import OVStableDiffusionPipeline

    cache = _cache_path(model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _say(f"loading model/export cache at {cache}")
    with FileLock(str(cache) + ".lock"):
        if not _valid_export(cache, model):
            if cache.exists():
                shutil.rmtree(cache)
            _say(f"exporting Stable Diffusion model {model}; first use downloads several GB")
            temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}-", dir=cache.parent))
            try:
                pipe = OVStableDiffusionPipeline.from_pretrained(
                    model, export=True, compile=False, device=device, safety_checker=None
                )
                pipe.save_pretrained(temporary)
                if not _valid_export_without_marker(temporary):
                    raise RuntimeError(
                        "model export is incomplete: required OpenVINO models or tokenizer/scheduler assets are missing"
                    )
                (temporary / ".openvino-image-complete.json").write_text(
                    json.dumps({"model": model, "schema": CACHE_SCHEMA})
                )
                os.replace(temporary, cache)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        else:
            _say("using cached OpenVINO export")
    return _load_cached(cache, device, img2img)


def _prepare_hub_pipeline(device: str, model: str, img2img: bool = False):
    from filelock import FileLock

    cache = _cache_path(model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache) + ".lock"):
        # Cache validation precedes every Hub operation, enabling true offline reuse.
        if _valid_export(cache, model):
            _say("using cached OpenVINO export")
            return _load_cached(cache, device, img2img)
        if cache.exists():
            shutil.rmtree(cache)
        _, files = _model_config(model)
        if _is_openvino_repository(files):
            _say(f"caching pre-exported OpenVINO model {model}")
            _publish_preexported(model, cache)
            return _load_cached(cache, device, img2img)
    # Original Diffusers repository: export via standard Optimum path and atomic app cache.
    return _default_pipeline(device, model, img2img)


def _validate_local_model(model_dir: Path) -> None:
    try:
        config = json.loads((model_dir / "model_index.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read local model configuration at {model_dir}: {exc}") from exc
    pipeline = config.get("_class_name")
    if pipeline not in SUPPORTED_PIPELINES:
        raise RuntimeError(
            f"Unsupported local model pipeline {pipeline!r}; expected a Diffusers StableDiffusionPipeline export"
        )
    if not _valid_export_without_marker(model_dir):
        raise RuntimeError(f"Local model at {model_dir} is not a complete OpenVINO Stable Diffusion export")


def _select_device(requested: str | None) -> str:
    import openvino as ov

    try:
        core = ov.Core()
        devices = list(core.available_devices)
    except Exception as exc:
        raise RuntimeError(f"Cannot detect OpenVINO devices; check the runtime installation: {exc}") from exc

    if requested is not None:
        if requested not in devices and not any(d.startswith(requested + ".") for d in devices):
            raise RuntimeError(
                f"OpenVINO device {requested} is unavailable; available devices: {', '.join(devices) or 'none'}"
            )
        _say(f"using explicitly requested device {requested}")
        return requested

    def cpu(reason: str) -> str:
        if "CPU" not in devices:
            raise RuntimeError(f"{reason}; CPU fallback is unavailable. Check OpenVINO device installation (available: {', '.join(devices) or 'none'})")
        _say(f"{reason}; selecting CPU")
        return "CPU"

    gpus = [d for d in devices if d == "GPU" or d.startswith("GPU.")]
    # Concrete IDs are necessary when OpenVINO enumerates multiple GPUs.
    if any(d.startswith("GPU.") for d in gpus):
        gpus = [d for d in gpus if d.startswith("GPU.")]
    integrated, discrete = [], []
    try:
        for gpu in gpus:
            name = core.get_property(gpu, ov.properties.device.full_name)
            kind = core.get_property(gpu, ov.properties.device.type)
            if "intel" not in name.lower():
                _say(f"ignoring non-Intel GPU {gpu}: {name}")
                continue
            if kind == ov.properties.device.Type.INTEGRATED:
                integrated.append(gpu)
            elif kind == ov.properties.device.Type.DISCRETE:
                discrete.append(gpu)
            else:
                return cpu(f"unknown GPU architecture for {gpu}")
    except Exception as exc:
        return cpu(f"GPU architecture detection failed ({exc})")

    npus = [d for d in devices if d == "NPU" or d.startswith("NPU.")]
    if discrete:
        selected, reason = discrete[0], "discrete Intel GPU detected"
    elif npus:
        selected = npus[0]
        reason = "integrated Intel GPU detected" if integrated else "no Intel GPU detected"
        reason += "; NPU available"
    elif integrated:
        selected, reason = integrated[0], "integrated Intel GPU detected; NPU unavailable"
    else:
        return cpu("no Intel GPU or NPU detected")
    _say(f"{reason}; selecting {selected}")
    return selected


def _default_model(device: str) -> str:
    return "OpenVINO/stable-diffusion-v1-5-int8-ov" if device.split(".")[0] == "NPU" else MODEL


def get_info(device: str | None, model_dir: Path | None, model: str | None) -> dict[str, str]:
    """Describe selection without contacting the Hub or loading/compiling a model."""
    selected = _select_device(device)
    return {
        "device": selected,
        "model": str(model_dir.resolve()) if model_dir is not None else (model or _default_model(selected)),
    }


def generate_prompts(
    prompts: list[str],
    device: str | None,
    model_dir: Path | None,
    model: str | None,
    seed: int | None,
    steps: int,
    reference=None,
    strength: float = 0.75,
    *,
    style=None,
):
    device = _select_device(device)
    if style is not None:
        if model_dir is not None or model != "stable-diffusion-v1-5/stable-diffusion-v1-5" or reference is not None:
            raise RuntimeError("Style conditioning requires the original SD 1.5 model, no --model-dir, and no img2img reference")
        model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    else:
        model = model or _default_model(device)
    import torch
    pipeline_class = _pipeline_class(reference is not None)

    style_runtime = None
    if style is not None:
        from . import style as style_module
        style_cache = style_module.prepare_style(device)
        pipe = _pipeline_class().from_pretrained(
            str(style_cache), export=False, compile=False, device=device, local_files_only=True, safety_checker=None
        )
        if device == "NPU" or device.startswith("NPU."):
            # Reshape standard OV components first; the adapter graph has its own proven fixed shape.
            pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
        try:
            style_runtime = style_module.StyleRuntime(pipe, style_cache, device)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to compile style encoder/adapter UNet for {device}: {exc}") from exc
        pipe.unet = style_runtime
    elif model_dir is not None:
        _say(f"loading local model from {model_dir}")
        _validate_local_model(model_dir)
        if reference is not None and not all((model_dir / "vae_encoder" / f"openvino_model.{ext}").is_file() for ext in ("xml", "bin")):
            raise RuntimeError(f"Image-to-image requires an encoder-bearing OpenVINO export with vae_encoder at {model_dir}; use a compatible --model or --model-dir")
        pipe = pipeline_class.from_pretrained(
            str(model_dir), export=False, compile=False, device=device, local_files_only=True, safety_checker=None
        )
    else:
        pipe = _prepare_hub_pipeline(device, model, img2img=True) if reference is not None else _prepare_hub_pipeline(device, model)
    pipe.set_progress_bar_config(disable=True)
    npu = device == "NPU" or device.startswith("NPU.")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    try:
        if npu and style_runtime is None:
            pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
        selected_model = str(model_dir.resolve()) if model_dir is not None else model
        shape_policy = ("style-" if style_runtime is not None else "") + ("img2img-" if reference is not None else "") + ("npu-static-1x512x512" if npu else "dynamic-512x512")
        if style_runtime is None:
            _configure_compiled_cache(pipe, selected_model, device, shape_policy)
        else:
            _say("style graphs compiled directly; device compiler-cache persistence is unavailable")
        _say(f"compiling for {device}")
        pipe.compile()
    except RuntimeError as exc:
        selected_model = str(model_dir) if model_dir is not None else model
        raise RuntimeError(
            f"Failed to prepare or compile model {selected_model!r} for {device}: {exc}"
        ) from exc
    for prompt in prompts:
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "num_images_per_prompt": 1,
        }
        if reference is None:
            kwargs["height"] = 512
            kwargs["width"] = 512
        else:
            kwargs["image"] = reference.copy()
            kwargs["strength"] = strength
        if style_runtime is not None:
            kwargs["ip_adapter_image_embeds"] = [style_runtime.embeds(style, pipe.feature_extractor)]
        if generator is not None:
            kwargs["generator"] = generator
        _say("generating image")
        images = pipe(**kwargs).images
        if len(images) != 1:
            raise RuntimeError(f"backend returned {len(images)} images for one prompt")
        yield images[0]


def generate(
    prompt: str,
    device: str | None,
    model_dir: Path | None,
    model: str | None,
    seed: int | None,
    steps: int,
    reference=None,
    strength: float = 0.75,
    *,
    style=None,
):
    return next(generate_prompts([prompt], device, model_dir, model, seed, steps, reference, strength, style=style))
