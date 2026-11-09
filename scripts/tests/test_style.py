import json
from pathlib import Path

import numpy as np
import pytest

from intel_image_generator import style


def test_style_export_requires_complete_graphs_but_not_safety_assets(tmp_path):
    required = (
        "unet/openvino_model.xml", "unet/openvino_model.bin",
        "text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin",
        "vae_decoder/openvino_model.xml", "vae_decoder/openvino_model.bin",
        "style_image_encoder/openvino_model.xml", "style_image_encoder/openvino_model.bin",
        "feature_extractor/preprocessor_config.json", "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json", "model_index.json",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    (tmp_path / "model_index.json").write_text(json.dumps({"safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"]}))
    marker = {"schema": style.SCHEMA, "base": style.BASE_REVISION,
              "adapter": style.ADAPTER_REVISION, "adapter_sha256": style.ADAPTER_SHA256,
              "encoder_sha256": style.ENCODER_SHA256, "scale": style.SCALE}
    (tmp_path / ".style-complete.json").write_text(json.dumps(marker))
    assert style._valid(tmp_path)
    assert not (tmp_path / "safety_checker").exists()
    (tmp_path / "style_image_encoder/openvino_model.bin").unlink()
    assert not style._valid(tmp_path)


@pytest.mark.parametrize("missing", [
    "text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin",
    "vae_decoder/openvino_model.xml", "vae_decoder/openvino_model.bin",
    "tokenizer/tokenizer_config.json", "scheduler/scheduler_config.json",
])
def test_style_export_rejects_missing_base_runtime_component(tmp_path, missing):
    required = (
        "unet/openvino_model.xml", "unet/openvino_model.bin",
        "text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin",
        "vae_decoder/openvino_model.xml", "vae_decoder/openvino_model.bin",
        "style_image_encoder/openvino_model.xml", "style_image_encoder/openvino_model.bin",
        "safety_checker/model.safetensors", "safety_checker/config.json",
        "feature_extractor/preprocessor_config.json", "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json", "model_index.json",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    (tmp_path / "model_index.json").write_text(json.dumps({"safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"]}))
    marker = {"schema": style.SCHEMA, "base": style.BASE_REVISION,
              "adapter": style.ADAPTER_REVISION, "adapter_sha256": style.ADAPTER_SHA256,
              "encoder_sha256": style.ENCODER_SHA256, "scale": style.SCALE}
    (tmp_path / ".style-complete.json").write_text(json.dumps(marker))
    (tmp_path / missing).unlink()
    assert not style._valid(tmp_path)


def test_style_loads_adapter_before_validating_encoder_and_processor(tmp_path):
    from types import SimpleNamespace

    class Processor:
        size = {"shortest_edge": 224}
        crop_size = {"height": 224, "width": 224}
        do_center_crop = True
        do_normalize = True
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

    class FakePipeline:
        def __init__(self):
            self.image_encoder = None
            self.feature_extractor = Processor()
            self.loaded = False
            self.scaled = False
        def load_ip_adapter(self, *args, **kwargs):
            assert self.image_encoder is None
            self.image_encoder = SimpleNamespace(config=SimpleNamespace(
                hidden_size=1280, projection_dim=1024, image_size=224, patch_size=14))
            self.loaded = True
        def set_ip_adapter_scale(self, value):
            assert self.loaded and value == style.SCALE
            self.scaled = True

    pipe = FakePipeline()
    style._load_adapter(pipe, tmp_path / "models/ip-adapter.safetensors", tmp_path / "encoder")
    assert pipe.loaded and pipe.scaled and pipe.image_encoder is not None


def _write_source_snapshot(path):
    for relative in style.BASE_ALLOW_PATTERNS:
        if "*" in relative:
            continue
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("asset")
    return path


def test_base_snapshot_partial_cache_fetches_only_required_safe_files(tmp_path, monkeypatch):
    class CacheMiss(Exception): pass
    hub = tmp_path / "hub"
    partial = hub / f"models--{style.BASE_MODEL.replace('/', '--')}" / "snapshots" / style.BASE_REVISION
    partial.mkdir(parents=True)
    (partial / "model_index.json").write_text("{}")
    monkeypatch.setattr(style, "_hub_cache_root", lambda: hub)
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        if kwargs["local_files_only"]:
            raise CacheMiss()
        return _write_source_snapshot(tmp_path / "online-snapshot")
    result = style._base_snapshot(download, CacheMiss, offline=False)
    assert result == tmp_path / "online-snapshot"
    assert len(calls) == 2 and calls[1]["allow_patterns"] == style.BASE_ALLOW_PATTERNS
    assert style._source_snapshot_complete(result)
    assert all("*.ckpt" not in pattern and "*.bin" not in pattern for pattern in style.BASE_ALLOW_PATTERNS)


def test_base_snapshot_partial_returned_from_local_attempt_triggers_scoped_online_fetch(tmp_path, monkeypatch):
    class CacheMiss(Exception): pass
    monkeypatch.setattr(style, "_hub_cache_root", lambda: tmp_path / "hub")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "model_index.json").write_text("{}")
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        return partial if kwargs["local_files_only"] else _write_source_snapshot(tmp_path / "complete")
    result = style._base_snapshot(download, CacheMiss, offline=False)
    assert result == tmp_path / "complete"
    assert [call["local_files_only"] for call in calls] == [True, False]
    assert all(call["allow_patterns"] == style.BASE_ALLOW_PATTERNS for call in calls)


def test_base_snapshot_falls_back_only_after_local_cache_miss_and_filters_pickle_assets(tmp_path, monkeypatch):
    class CacheMiss(Exception): pass
    monkeypatch.setattr(style, "_hub_cache_root", lambda: tmp_path / "hub")
    calls = []
    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        if len(calls) == 1:
            raise CacheMiss()
        return _write_source_snapshot(tmp_path / "downloaded")
    assert style._base_snapshot(download, CacheMiss, offline=False) == tmp_path / "downloaded"
    assert len(calls) == 2
    assert calls[0][1]["local_files_only"] is True
    assert calls[1][1]["local_files_only"] is False
    assert calls[0][1]["allow_patterns"] == style.BASE_ALLOW_PATTERNS
    assert not any(pattern.endswith("*.ckpt") or pattern.endswith("*.bin") for pattern in style.BASE_ALLOW_PATTERNS)


def test_cold_pinned_base_export_disables_safety_and_uses_pinned_source(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "_pinned_base_export_path", lambda: tmp_path / "pinned-cache")
    calls = []
    class FakePipe:
        def save_pretrained(self, directory):
            root = Path(directory)
            for relative in (
                "model_index.json", "unet/openvino_model.xml", "unet/openvino_model.bin",
                "text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin",
                "vae_decoder/openvino_model.xml", "vae_decoder/openvino_model.bin",
                "tokenizer/tokenizer_config.json", "scheduler/scheduler_config.json",
                "feature_extractor/preprocessor_config.json",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"]}) if relative == "model_index.json" else "asset")
    class FakePipelineClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return FakePipe()
    result = style._ensure_pinned_base_export(tmp_path / "immutable-snapshot", "NPU", FakePipelineClass)
    assert calls == [(str(tmp_path / "immutable-snapshot"), {
        "export": True, "compile": False, "device": "NPU", "local_files_only": True, "safety_checker": None,
    })]
    assert style._valid_pinned_base(result)
    assert not (result / "safety_checker").exists()


def test_pinned_base_cache_identity_changes_with_revision(monkeypatch):
    original = style._pinned_base_export_path()
    monkeypatch.setattr(style, "BASE_REVISION", "a" * 40)
    assert style._pinned_base_export_path() != original


def test_base_snapshot_offline_cache_miss_does_not_try_online(tmp_path, monkeypatch):
    class CacheMiss(Exception): pass
    monkeypatch.setattr(style, "_hub_cache_root", lambda: tmp_path / "hub")
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        raise CacheMiss()
    with pytest.raises(RuntimeError, match="incomplete in the local cache"):
        style._base_snapshot(download, CacheMiss, offline=True)
    assert len(calls) == 1


def test_style_export_marker_rejects_changed_asset_identity(tmp_path):
    (tmp_path / ".style-complete.json").write_text(json.dumps({"schema": style.SCHEMA}))
    assert not style._valid(tmp_path)


def test_standard_adapter_cfg_embedding_order_and_shape():
    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images == "reference" and return_tensors == "np"
            return type("Batch", (), {"pixel_values": np.ones((1, 3, 224, 224), dtype=np.float32)})()

    class Compiled:
        inputs = [type("Input", (), {"get_names": lambda self: {"pixel_values"}})()]
        def __call__(self, inputs):
            assert inputs["pixel_values"].shape == (1, 3, 224, 224)
            return {"image_embeds": np.full((1, 1024), 0.25, dtype=np.float32)}

    class Torch:
        @staticmethod
        def from_numpy(value): return value

    runtime = object.__new__(style.StyleRuntime)
    runtime.encoder = Compiled()
    runtime.torch = Torch()
    result = runtime.embeds("reference", Processor())
    assert result.shape == (2, 1, 1024)
    assert np.all(result[0] == 0)
    assert np.all(result[1] == 0.25)
