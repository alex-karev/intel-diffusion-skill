"""Standard SD 1.5 IP-Adapter export and OpenVINO runtime bridge."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
ADAPTER_REPO = "h94/IP-Adapter"
ADAPTER_REVISION = "3cf3eb82240920f904900a11faf2e1b62ea3b610"
ADAPTER_FILE = "models/ip-adapter_sd15.safetensors"
ENCODER_FILE = "models/image_encoder/model.safetensors"
SCALE = 0.7
ADAPTER_SHA256 = "289b45f16d043d0bf542e45831f971dcdaabe18b656f11e86d9dfba7e9ee3369"
ENCODER_SHA256 = "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030"
SCHEMA = 3
BASE_ALLOW_PATTERNS = (
    "model_index.json", "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "feature_extractor/preprocessor_config.json",
)


def _versions() -> dict[str, str]:
    return {p: importlib.metadata.version(p) for p in ("openvino", "optimum-intel", "diffusers", "transformers", "torch")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style_cache_path() -> Path:
    ident = [BASE_MODEL, BASE_REVISION, ADAPTER_REPO, ADAPTER_REVISION, ADAPTER_FILE, ENCODER_FILE, SCALE, SCHEMA, _versions(), "fp32", "static-cfg2-512"]
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:24]
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "intel-image-generator" / "style" / key


def _hub_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface" / "hub"


def _source_snapshot_complete(path: Path) -> bool:
    required = (
        "model_index.json", "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
        "tokenizer/vocab.json", "tokenizer/merges.txt",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "feature_extractor/preprocessor_config.json",
    )
    return all((path / item).is_file() for item in required)


def _base_snapshot(snapshot_download, local_entry_not_found_error, *, offline: bool) -> Path:
    cached = _hub_cache_root() / f"models--{BASE_MODEL.replace('/', '--')}" / "snapshots" / BASE_REVISION
    if _source_snapshot_complete(cached):
        return cached
    options = {"revision": BASE_REVISION, "allow_patterns": BASE_ALLOW_PATTERNS}
    try:
        snapshot = Path(snapshot_download(BASE_MODEL, local_files_only=True, **options))
    except local_entry_not_found_error as exc:
        if offline:
            raise RuntimeError(f"Pinned base revision {BASE_REVISION} is incomplete in the local cache") from exc
        snapshot = Path(snapshot_download(BASE_MODEL, local_files_only=False, **options))
    if not _source_snapshot_complete(snapshot):
        if offline:
            raise RuntimeError(f"Pinned base snapshot is incomplete at {snapshot}; required safe Diffusers files are missing")
        snapshot = Path(snapshot_download(BASE_MODEL, local_files_only=False, **options))
    if not _source_snapshot_complete(snapshot):
        raise RuntimeError(f"Pinned base snapshot is incomplete at {snapshot}; required safe Diffusers files are missing")
    return snapshot


def _pinned_base_export_path() -> Path:
    ident = [BASE_MODEL, BASE_REVISION, SCHEMA, _versions(), "fp32", "optimum-openvino"]
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:24]
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "intel-image-generator" / "pinned-base" / key


def _valid_pinned_base(path: Path) -> bool:
    from .generation import _valid_export_without_marker
    try:
        marker = json.loads((path / ".pinned-base-complete.json").read_text())
    except (OSError, ValueError):
        return False
    expected = {"model": BASE_MODEL, "revision": BASE_REVISION, "schema": SCHEMA}
    return (marker == expected and _valid_export_without_marker(path)
            and (path / "feature_extractor/preprocessor_config.json").is_file())


def _ensure_pinned_base_export(source: Path, device: str, pipeline_class=None) -> Path:
    from filelock import FileLock
    if pipeline_class is None:
        from optimum.intel.openvino import OVStableDiffusionPipeline
        pipeline_class = OVStableDiffusionPipeline

    cache = _pinned_base_export_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache) + ".lock"):
        if _valid_pinned_base(cache):
            return cache
        if cache.exists():
            shutil.rmtree(cache)
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}-", dir=cache.parent))
        try:
            pipe = pipeline_class.from_pretrained(
                str(source), export=True, compile=False, device=device, local_files_only=True, safety_checker=None
            )
            pipe.save_pretrained(temporary)
            if not _valid_export_components(temporary):
                raise RuntimeError("Pinned base OpenVINO export is incomplete or lacks its image processor")
            (temporary / ".pinned-base-complete.json").write_text(json.dumps({
                "model": BASE_MODEL, "revision": BASE_REVISION, "schema": SCHEMA,
            }))
            if not _valid_pinned_base(temporary):
                raise RuntimeError("Pinned base export failed required-component validation")
            os.replace(temporary, cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return cache


def _valid_export_components(path: Path) -> bool:
    from .generation import _valid_export_without_marker
    return (_valid_export_without_marker(path)
            and (path / "feature_extractor/preprocessor_config.json").is_file())


def _valid(path: Path) -> bool:
    try:
        marker = json.loads((path / ".style-complete.json").read_text())
    except (OSError, ValueError):
        return False
    expected = {"schema": SCHEMA, "base": BASE_REVISION, "adapter": ADAPTER_REVISION,
                "adapter_sha256": ADAPTER_SHA256, "encoder_sha256": ENCODER_SHA256, "scale": SCALE}
    components = (
        "unet/openvino_model.xml", "unet/openvino_model.bin",
        "text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin",
        "vae_decoder/openvino_model.xml", "vae_decoder/openvino_model.bin",
        "style_image_encoder/openvino_model.xml", "style_image_encoder/openvino_model.bin",
        "feature_extractor/preprocessor_config.json", "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json", "model_index.json",
    )
    from .generation import _valid_export_without_marker
    return (marker == expected and _valid_export_without_marker(path)
            and all((path / item).is_file() for item in components))


def _load_adapter(pipe, adapter_path: Path, encoder_path: Path) -> None:
    """Load native IP-Adapter weights before validating the normally absent encoder component."""
    pipe.load_ip_adapter(str(adapter_path.parent.parent), subfolder="models", weight_name=adapter_path.name,
                         image_encoder_pretrained_model_name_or_path=str(encoder_path.parent),
                         low_cpu_mem_usage=False)
    encoder = getattr(pipe, "image_encoder", None)
    config = getattr(encoder, "config", None)
    if (getattr(config, "hidden_size", None) != 1280
            or getattr(config, "projection_dim", None) != 1024
            or getattr(config, "image_size", None) != 224
            or getattr(config, "patch_size", None) != 14):
        raise RuntimeError("Pinned adapter encoder config is not CLIP ViT-H/14 with 1024-D pooled embeddings")
    processor = getattr(pipe, "feature_extractor", None)
    if processor is None:
        raise RuntimeError("Base model has no CLIP image processor; refusing style export")
    size = processor.size
    crop = processor.crop_size
    size_value = size.get("shortest_edge") if isinstance(size, dict) else size
    crop_value = crop.get("height") if isinstance(crop, dict) else crop
    mean = tuple(float(v) for v in processor.image_mean)
    std = tuple(float(v) for v in processor.image_std)
    expected_mean = (0.48145466, 0.4578275, 0.40821073)
    expected_std = (0.26862954, 0.26130258, 0.27577711)
    if (size_value != 224 or crop_value != 224 or not processor.do_center_crop
            or not processor.do_normalize or len(mean) != 3 or len(std) != 3
            or any(abs(a - b) > 1e-6 for a, b in zip(mean, expected_mean, strict=True))
            or any(abs(a - b) > 1e-6 for a, b in zip(std, expected_std, strict=True))):
        raise RuntimeError("Base image processor is incompatible with the pinned CLIP ViT-H/14 preprocessing contract")
    pipe.set_ip_adapter_scale(SCALE)


def _build(cache: Path, device: str) -> None:
    """Export native Diffusers IP-Adapter operations into a separate OV model cache."""
    import openvino as ov
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from huggingface_hub.errors import LocalEntryNotFoundError

    base = _base_snapshot(snapshot_download, LocalEntryNotFoundError, offline=HF_HUB_OFFLINE)
    adapter_root = _hub_cache_root() / "models--h94--IP-Adapter" / "snapshots" / ADAPTER_REVISION
    adapter_local, encoder_local = adapter_root / ADAPTER_FILE, adapter_root / ENCODER_FILE
    if adapter_local.is_file() and encoder_local.is_file():
        adapter_path, encoder_path = adapter_local, encoder_local
    else:
        adapter_path = Path(hf_hub_download(ADAPTER_REPO, ADAPTER_FILE, revision=ADAPTER_REVISION, local_files_only=HF_HUB_OFFLINE))
        encoder_path = Path(hf_hub_download(ADAPTER_REPO, ENCODER_FILE, revision=ADAPTER_REVISION, local_files_only=HF_HUB_OFFLINE))
    for path, expected, label in ((adapter_path, ADAPTER_SHA256, "adapter"), (encoder_path, ENCODER_SHA256, "image encoder")):
        if _sha256(path) != expected:
            raise RuntimeError(f"Pinned IP-Adapter {label} checksum mismatch at {path}")
    # Export a separate base cache keyed by immutable revision; never trust a mutable-ID cache.
    base_export = _ensure_pinned_base_export(base, device)
    from .generation import _say
    _say(f"style export: loading pinned SD 1.5 source from {base}")
    pipe = StableDiffusionPipeline.from_pretrained(str(base), local_files_only=True, torch_dtype=torch.float32, safety_checker=None)
    _load_adapter(pipe, adapter_path, encoder_path)

    class UnetGraph(torch.nn.Module):
        def __init__(self, unet):
            super().__init__(); self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states, image_embeds):
            return self.unet(sample, timestep, encoder_hidden_states,
                             added_cond_kwargs={"image_embeds": [image_embeds]}, return_dict=False)[0]

    latent = torch.zeros((2, 4, 64, 64)); timestep = torch.tensor(500.0)
    text = torch.zeros((2, 77, 768)); embeds = torch.zeros((2, 1, 1024))
    _say("style export: converting adapter-aware UNet to OpenVINO")
    unet_model = ov.convert_model(UnetGraph(pipe.unet).eval(), example_input=(latent, timestep, text, embeds))
    # Fixed shape is explicit: one prompt with CFG, 512x512 generation.
    unet_model.reshape({"sample": [2, 4, 64, 64], "timestep": [], "encoder_hidden_states": [2, 77, 768], "image_embeds": [2, 1, 1024]})
    class EncoderGraph(torch.nn.Module):
        def __init__(self, encoder): super().__init__(); self.encoder = encoder
        def forward(self, pixel_values): return self.encoder(pixel_values).image_embeds
    pixels = torch.zeros((1, 3, 224, 224))
    _say("style export: converting ViT-H image encoder to OpenVINO")
    image_model = ov.convert_model(EncoderGraph(pipe.image_encoder).eval(), example_input=(pixels,))
    image_input_name = next(iter(image_model.inputs[0].get_names()))
    image_model.reshape({image_input_name: [1, 3, 224, 224]})
    _say("style export: publishing complete graphs beside the base export")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}-", dir=cache.parent))
    try:
        shutil.copytree(base_export, temporary, dirs_exist_ok=True)
        ov.save_model(unet_model, str(temporary / "unet" / "openvino_model.xml"))
        (temporary / "style_image_encoder").mkdir()
        ov.save_model(image_model, str(temporary / "style_image_encoder" / "openvino_model.xml"))
        (temporary / ".style-complete.json").write_text(json.dumps({"schema": SCHEMA, "base": BASE_REVISION, "adapter": ADAPTER_REVISION, "adapter_sha256": ADAPTER_SHA256, "encoder_sha256": ENCODER_SHA256, "scale": SCALE}))
        if not _valid(temporary):
            raise RuntimeError("Style export failed completeness validation")
        os.replace(temporary, cache)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def prepare_style(device: str):
    from filelock import FileLock
    cache = style_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache) + ".lock"):
        if not _valid(cache):
            if cache.exists(): shutil.rmtree(cache)
            try: _build(cache, device)
            except Exception as exc:
                if os.environ.get("HF_HUB_OFFLINE") == "1":
                    raise RuntimeError(f"Offline style export is incomplete at {cache}: {exc}") from exc
                raise
    if not _valid(cache): raise RuntimeError(f"Incomplete style export at {cache}")
    return cache


class StyleRuntime:
    """OpenVINO encoder and adapter-aware UNet bridge used by the Diffusers loop."""
    def __init__(self, pipe, cache: Path, device: str):
        import openvino as ov
        import torch
        from types import SimpleNamespace
        self.torch = torch
        self.config = pipe.unet.config
        self.device = device
        core = ov.Core()
        from .generation import _say
        _say(f"style runtime: compiling image encoder for {device}")
        self.encoder = core.compile_model(str(cache / "style_image_encoder" / "openvino_model.xml"), device)
        _say(f"style runtime: image encoder compiled for {device}")
        unet_model = core.read_model(str(cache / "unet" / "openvino_model.xml"))
        _say(f"style runtime: compiling adapter UNet for {device}")
        self.unet = core.compile_model(unet_model, device)
        _say(f"style runtime: adapter UNet compiled for {device}")
        # Diffusers uses this only to determine adapter count for raw image inputs; we pass precomputed embeddings.
        self.encoder_hid_proj = SimpleNamespace(image_projection_layers=[object()])
        self.dtype = torch.float32

    def embeds(self, image, processor):
        import numpy as np
        values = processor(images=image, return_tensors="np").pixel_values
        result = self.encoder({next(iter(self.encoder.inputs[0].get_names())): values})
        positive = next(iter(result.values()))
        if positive.shape != (1, 1024):
            raise RuntimeError(f"Unexpected IP-Adapter image embedding shape {positive.shape}; expected (1, 1024)")
        # Standard pooled IP-Adapter uses zero unconditional embeddings for CFG.
        embeddings = np.concatenate([np.zeros_like(positive), positive], axis=0)[:, None, :]
        return self.torch.from_numpy(embeddings)

    def __call__(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, **kwargs):
        image_embeds = added_cond_kwargs["image_embeds"][0]
        torch = self.torch
        inputs = {"sample": sample, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states, "image_embeds": image_embeds}
        result = self.unet(inputs)
        output = torch.from_numpy(next(iter(result.values())))
        return (output,)

    def reshape(self, **kwargs):
        # The adapter UNet graph was already exported with the fixed CFG/512 policy.
        return self

    def compile(self): return self
