import json
import sys
import types
from pathlib import Path

import pytest

from intel_image_generator import generation


class FakePipe:
    def __init__(self):
        self.calls = []
        self.last_kwargs = None
    def reshape(self, **kwargs):
        self.calls.append(("reshape", kwargs))
    def compile(self):
        self.calls.append(("compile",))
    def set_progress_bar_config(self, **kwargs):
        self.calls.append(("progress", kwargs))
    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return types.SimpleNamespace(images=["generated"])
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for component in ("unet", "text_encoder", "vae_decoder"):
            directory = path / component
            directory.mkdir(exist_ok=True)
            (directory / "openvino_model.xml").touch()
            (directory / "openvino_model.bin").touch()
        for component, name in (("tokenizer", "tokenizer_config.json"), ("scheduler", "scheduler_config.json")):
            directory = path / component
            directory.mkdir(exist_ok=True)
            (directory / name).touch()
        (path / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))


def install_fakes(monkeypatch, devices=("GPU",)):
    pipe, calls = FakePipe(), []
    class Core:
        available_devices = list(devices)
    class Pipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return pipe
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    for name in ("optimum", "optimum.intel", "optimum.intel.openvino"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["optimum.intel.openvino"].OVStableDiffusionPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=lambda: types.SimpleNamespace(manual_seed=lambda seed: ("generator", seed))))
    return pipe, calls


def seed_cache(pipe, path, model=generation.MODEL):
    path.mkdir(parents=True)
    pipe.save_pretrained(path)
    (path / ".openvino-image-complete.json").write_text(json.dumps({"model": model, "schema": generation.CACHE_SCHEMA}))


def install_hub(monkeypatch, config, files=()):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda model, filename: (calls.append(("config", model, filename)) or str(_write_config(config)))
    hub.list_repo_files = lambda model: (calls.append(("files", model)) or list(files))
    hub.snapshot_download = lambda model, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected snapshot download"))
    errors = types.ModuleType("huggingface_hub.errors")
    for name in ("EntryNotFoundError", "GatedRepoError", "HfHubHTTPError", "HFValidationError", "LocalEntryNotFoundError", "RepositoryNotFoundError"):
        setattr(errors, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return calls


def _write_config(config):
    import tempfile
    path = Path(tempfile.mkstemp()[1])
    path.write_text(json.dumps(config))
    return path


def seed_hub_export(path, model, class_name="StableDiffusionPipeline"):
    path.mkdir(parents=True)
    pipe = FakePipe()
    pipe.save_pretrained(path)
    (path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    (path / ".openvino-image-complete.json").write_text(json.dumps({"model": model, "schema": generation.CACHE_SCHEMA}))


@pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
def test_sequential_prompts_share_load_compile_and_seeded_generator(monkeypatch, device):
    pipe, _ = install_fakes(monkeypatch, (device,))
    calls = []
    monkeypatch.setattr(FakePipe, "__call__", lambda self, **kwargs: (calls.append(kwargs) or types.SimpleNamespace(images=[kwargs["prompt"]])))
    loaded = []
    monkeypatch.setattr(generation, "_prepare_hub_pipeline", lambda device, model: loaded.append(model) or pipe)
    assert list(generation.generate_prompts([str(i) for i in range(9)], device, None, "org/model", 42, 3)) == list(map(str, range(9)))
    assert len(loaded) == 1
    assert [item["prompt"] for item in calls] == list(map(str, range(9)))
    assert len({id(item["generator"]) for item in calls}) == 1
    assert [item[0] for item in pipe.calls if item[0] in {"reshape", "compile"}] == (["reshape", "compile"] if device == "NPU" else ["compile"])
    if device == "NPU":
        assert next(call[1] for call in pipe.calls if call[0] == "reshape") == {"batch_size": 1, "height": 512, "width": 512, "num_images_per_prompt": 1}
    assert sum(item[0] == "compile" for item in pipe.calls) == 1


def test_sequential_backend_rejects_image_count_mismatch(monkeypatch):
    pipe, _ = install_fakes(monkeypatch, ("GPU",))
    monkeypatch.setattr(generation, "_prepare_hub_pipeline", lambda device, model: pipe)
    monkeypatch.setattr(FakePipe, "__call__", lambda self, **kwargs: types.SimpleNamespace(images=[]))
    with pytest.raises(RuntimeError, match="returned 0 images for one prompt"):
        list(generation.generate_prompts(["one"], "GPU", None, "org/model", None, 2))


def test_gpu_missing_before_pipeline_import(monkeypatch):
    class Core:
        available_devices = ["CPU"]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.delitem(sys.modules, "optimum.intel.openvino", raising=False)
    with pytest.raises(RuntimeError, match="GPU is unavailable"):
        generation.generate("x", "GPU", None, None, None, 2)


def test_legacy_default_cache_loaded_without_hub_lookup(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    cache = tmp_path / "openvino-image" / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    monkeypatch.setattr(generation, "_model_config", lambda model: (_ for _ in ()).throw(AssertionError("Hub config queried")))
    assert generation.generate("prompt", "GPU", None, None, 17, 4) == "generated"
    assert calls == [(str(cache), {"export": False, "compile": False, "device": "GPU", "local_files_only": True})]
    assert pipe.last_kwargs == {"prompt": "prompt", "num_inference_steps": 4, "num_images_per_prompt": 1, "height": 512, "width": 512, "generator": ("generator", 17)}
    assert ("compile",) in pipe.calls


def test_first_export_then_cache_hit(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    cache = tmp_path / "openvino-image" / "cache"
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    generation._default_pipeline("GPU", generation.MODEL)
    generation._default_pipeline("GPU", generation.MODEL)
    assert len(calls) == 3
    assert calls[0][0] == generation.MODEL and calls[0][1]["export"] is True
    assert calls[1][0] == str(cache) and calls[1][1]["local_files_only"] is True
    assert calls[2][0] == str(cache)
    assert (cache / ".openvino-image-complete.json").is_file()
    assert not list(cache.parent.glob(".cache-*"))


def test_incomplete_cache_discarded_and_rebuilt(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "partial").touch()
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    generation._default_pipeline("GPU", generation.MODEL)
    assert calls[0][1]["export"] is True
    assert calls[1][0] == str(cache)
    assert (cache / ".openvino-image-complete.json").is_file()
    assert not (cache / "partial").exists()


def test_export_failure_removes_temporary_cache(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    cache = tmp_path / "cache"
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    monkeypatch.setattr(pipe, "save_pretrained", lambda path: (_ for _ in ()).throw(RuntimeError("export failed")))
    with pytest.raises(RuntimeError, match="export failed"):
        generation._default_pipeline("GPU", generation.MODEL)
    assert not cache.exists()
    assert not list(cache.parent.glob(".cache-*"))
    assert calls[0][1]["export"] is True


def test_local_model_loaded_offline_and_safety_options_untouched(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch, ("CPU", "GPU"))
    model_dir = tmp_path / "model"
    pipe.save_pretrained(model_dir)
    generation.generate("x", "CPU", model_dir, None, None, 2)
    assert calls[0] == (str(model_dir), {"export": False, "compile": False, "device": "CPU", "local_files_only": True})
    assert "safety_checker" not in calls[0][1]
    assert ("compile",) in pipe.calls


def test_local_model_unsupported_pipeline_rejected_before_load(monkeypatch, tmp_path):
    _, calls = install_fakes(monkeypatch)
    model_dir = tmp_path / "bad"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionInpaintPipeline"}))
    with pytest.raises(RuntimeError, match="Unsupported local model pipeline"):
        generation.generate("x", "GPU", model_dir, None, None, 2)
    assert calls == []


def test_compiled_cache_supported_sets_shared_option_and_preserves_settings(monkeypatch, tmp_path, capsys):
    class Core:
        def get_property(self, device, prop):
            return {"OPTIMIZATION_CAPABILITIES": ["EXPORT_IMPORT"], "SUPPORTED_PROPERTIES": ["CACHE_DIR"]}[prop]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    component = types.SimpleNamespace(ov_config={"NUM_STREAMS": "1"})
    pipe = types.SimpleNamespace(ov_config={"PERFORMANCE_HINT": "LATENCY"}, components={"unet": component})
    generation._configure_compiled_cache(pipe, "org/model", "GPU.0", "dynamic-512x512")
    cache = Path(pipe.ov_config["CACHE_DIR"])
    assert cache.is_dir()
    assert cache.parent == tmp_path / "intel-image-generator" / "compiled"
    assert pipe.ov_config["PERFORMANCE_HINT"] == "LATENCY"
    assert component.ov_config == {"NUM_STREAMS": "1", "CACHE_DIR": str(cache)}
    assert "cache hit not verified" in capsys.readouterr().err


@pytest.mark.parametrize("capabilities,properties", [([], ["CACHE_DIR"]), (["EXPORT_IMPORT"], [])])
def test_compiled_cache_unsupported_does_not_create_directory(monkeypatch, tmp_path, capabilities, properties):
    class Core:
        def get_property(self, device, prop):
            return {"OPTIMIZATION_CAPABILITIES": capabilities, "SUPPORTED_PROPERTIES": properties}[prop]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pipe = types.SimpleNamespace(ov_config={"PERFORMANCE_HINT": "LATENCY"})
    generation._configure_compiled_cache(pipe, "org/model", "GPU", "dynamic")
    assert pipe.ov_config == {"PERFORMANCE_HINT": "LATENCY"}
    assert not (tmp_path / "intel-image-generator").exists()


def test_compiled_cache_probe_failure_skips(monkeypatch, tmp_path, capsys):
    class Core:
        def get_property(self, *args):
            raise RuntimeError("probe failed")
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pipe = types.SimpleNamespace(ov_config={})
    generation._configure_compiled_cache(pipe, "org/model", "GPU", "dynamic")
    assert pipe.ov_config == {} and not (tmp_path / "intel-image-generator").exists()
    assert "probe failed" in capsys.readouterr().err


def test_compiled_cache_namespaces_model_device_version_shape_and_home(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(generation.importlib.metadata, "version", lambda package: "1.2")
    base = generation._compiled_cache_path("org/model", "GPU", "dynamic")
    assert base == generation._compiled_cache_path("org/model", "GPU", "dynamic")
    assert base != generation._compiled_cache_path("org/other", "GPU", "dynamic")
    assert base != generation._compiled_cache_path("org/model", "CPU", "dynamic")
    assert base != generation._compiled_cache_path("org/model", "GPU", "npu-static")
    assert base.parent == tmp_path / ".cache" / "intel-image-generator" / "compiled"


def test_compiled_cache_local_path_namespace(monkeypatch, tmp_path):
    class Core:
        def get_property(self, device, prop):
            return {"OPTIMIZATION_CAPABILITIES": ["EXPORT_IMPORT"], "SUPPORTED_PROPERTIES": ["CACHE_DIR"]}[prop]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "model"
    pipe = types.SimpleNamespace(ov_config={})
    generation._configure_compiled_cache(pipe, str(path.resolve()), "CPU", "dynamic")
    assert pipe.ov_config["CACHE_DIR"].startswith(str(tmp_path / "cache"))


def test_custom_cache_separation(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(generation.importlib.metadata, "version", lambda _: "1")
    assert generation._cache_path("org/one") != generation._cache_path("org/two")
    assert generation._cache_path().parent == tmp_path / "openvino-image"


def test_cached_custom_export_offline_before_config(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    model = "org/custom"
    cache = tmp_path / "custom"
    seed_cache(pipe, cache, model)
    monkeypatch.setattr(generation, "_cache_path", lambda selected=generation.MODEL: cache)
    monkeypatch.setattr(generation, "_model_config", lambda selected: (_ for _ in ()).throw(AssertionError("network/config lookup")))
    assert generation._prepare_hub_pipeline("GPU", model) is pipe
    assert calls[0][0] == str(cache) and calls[0][1]["local_files_only"] is True


def test_original_model_config_is_queried_once_then_exported(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    model = "org/original"
    hub_calls = install_hub(monkeypatch, {"_class_name": "StableDiffusionPipeline"}, files=["model_index.json"])
    cache = tmp_path / "custom"
    monkeypatch.setattr(generation, "_cache_path", lambda selected=generation.MODEL: cache)
    generation._prepare_hub_pipeline("GPU", model)
    assert [call[0] for call in hub_calls] == ["config", "files"]
    assert calls[0][0] == model and calls[0][1]["export"] is True
    assert calls[-1][0] == str(cache)


def test_standard_class_preexported_repository_cached_and_loaded_locally(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch)
    model = "org/pre-exported"
    components = ("unet", "text_encoder", "vae_decoder")
    files = [f"{name}/openvino_model.{extension}" for name in components for extension in ("xml", "bin")]
    files += ["model_index.json", "tokenizer/tokenizer_config.json", "scheduler/scheduler_config.json"]
    hub_calls = install_hub(monkeypatch, {"_class_name": "StableDiffusionPipeline"}, files=files)
    cache = tmp_path / "cached-preexport"
    monkeypatch.setattr(generation, "_cache_path", lambda selected=generation.MODEL: cache)
    source = tmp_path / "hub-snapshot"
    FakePipe().save_pretrained(source)
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", lambda *args, **kwargs: str(source))
    assert generation._prepare_hub_pipeline("GPU", model) is pipe
    assert cache.is_dir() and generation._valid_export(cache, model)
    assert calls[0][0] == str(cache) and calls[0][1]["local_files_only"] is True
    assert [call[0] for call in hub_calls] == ["config", "files"]


def test_unsupported_hub_pipeline_rejected(monkeypatch):
    install_hub(monkeypatch, {"_class_name": "StableDiffusionInpaintPipeline"})
    with pytest.raises(RuntimeError, match="Unsupported model pipeline"):
        generation._model_config("org/inpaint")


def test_hub_access_errors_actionable_without_tokens(monkeypatch):
    calls = install_hub(monkeypatch, {"_class_name": "StableDiffusionPipeline"})
    hub = sys.modules["huggingface_hub"]
    errors = sys.modules["huggingface_hub.errors"]
    hub.hf_hub_download = lambda *args: (_ for _ in ()).throw(errors.GatedRepoError("gated"))
    with pytest.raises(RuntimeError, match="private or gated") as exc:
        generation._model_config("org/private")
    assert "token" not in str(exc.value).lower()


def test_npu_missing_before_pipeline_import(monkeypatch):
    class Core:
        available_devices = ["CPU", "GPU"]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.delitem(sys.modules, "optimum.intel.openvino", raising=False)
    with pytest.raises(RuntimeError, match="NPU is unavailable"):
        generation.generate("x", "NPU", None, generation.MODEL, None, 2)


def test_npu_static_reshape_before_compile_and_cache_reused_across_devices(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch, ("NPU", "GPU"))
    cache = tmp_path / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    monkeypatch.setattr(generation, "_model_config", lambda model: pytest.fail("unexpected Hub lookup"))
    generation.generate("x", "NPU", None, generation.MODEL, None, 2)
    assert [(c[0], c[1] if len(c) > 1 else None) for c in pipe.calls if c[0] in {"reshape", "compile"}] == [("reshape", {"batch_size": 1, "height": 512, "width": 512, "num_images_per_prompt": 1}), ("compile", None)]
    pipe.calls.clear()
    generation.generate("x", "GPU", None, None, None, 2)
    assert ("compile",) in pipe.calls
    assert [call[1]["device"] for call in calls] == ["NPU", "GPU"]
    assert all(call[1]["export"] is False for call in calls)


def test_npu_compile_error_identifies_model_and_device(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch, ("NPU",))
    cache = tmp_path / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    monkeypatch.setattr(pipe, "compile", lambda: (_ for _ in ()).throw(RuntimeError("compiler failed")))
    with pytest.raises(RuntimeError, match="model .* for NPU: compiler failed"): 
        generation.generate("x", "NPU", None, generation.MODEL, None, 2)
    assert len(calls) == 1 and calls[0][1]["device"] == "NPU"


@pytest.mark.parametrize("stage", ["compile", "inference"])
def test_npu_failure_preserves_output_without_fallback(stage, monkeypatch, tmp_path, capsys):
    from intel_image_generator import cli

    pipe, calls = install_fakes(monkeypatch, ("NPU", "GPU", "CPU"))
    cache = tmp_path / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    if stage == "compile":
        monkeypatch.setattr(pipe, "compile", lambda: (_ for _ in ()).throw(RuntimeError("compile failed")))
    else:
        monkeypatch.setattr(FakePipe, "__call__", lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("inference failed")))
    output = tmp_path / "output.png"
    output.write_bytes(b"original")
    monkeypatch.setattr(sys, "argv", ["intel-image-generator", "--device", "NPU", "--model", generation.MODEL, "-o", str(output), "-p", "prompt"])
    assert cli.main() == 1
    assert output.read_bytes() == b"original"
    assert f"{stage} failed" in capsys.readouterr().err
    assert len(calls) == 1 and calls[0][1]["device"] == "NPU"


def install_device_core(monkeypatch, devices, properties=None):
    properties = properties or {}
    device_properties = types.SimpleNamespace(
        full_name="name", type="type", Type=types.SimpleNamespace(INTEGRATED="integrated", DISCRETE="discrete")
    )
    class Core:
        available_devices = devices
        def get_property(self, device, prop):
            value = properties[(device, prop)]
            if isinstance(value, Exception):
                raise value
            return value
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(
        Core=Core, properties=types.SimpleNamespace(device=device_properties)
    ))


@pytest.mark.parametrize("devices,properties,expected", [
    (["CPU", "GPU", "NPU"], {("GPU", "name"): "Intel GPU", ("GPU", "type"): "integrated"}, "NPU"),
    (["CPU", "GPU"], {("GPU", "name"): "Intel GPU", ("GPU", "type"): "integrated"}, "GPU"),
    (["CPU", "GPU", "NPU"], {("GPU", "name"): "Intel GPU", ("GPU", "type"): "discrete"}, "GPU"),
    (["CPU"], {}, "CPU"),
    (["CPU", "NPU.0"], {}, "NPU.0"),
    (["CPU", "GPU.0", "GPU.1", "NPU"], {("GPU.0", "name"): "Intel GPU", ("GPU.0", "type"): "integrated", ("GPU.1", "name"): "Intel GPU", ("GPU.1", "type"): "discrete"}, "GPU.1"),
    (["CPU", "GPU", "NPU"], {("GPU", "name"): "Intel GPU", ("GPU", "type"): "unknown"}, "CPU"),
    (["CPU", "GPU", "NPU"], {("GPU", "name"): RuntimeError("probe error")}, "CPU"),
    (["CPU", "GPU", "NPU"], {("GPU", "name"): "Other GPU", ("GPU", "type"): "discrete"}, "NPU"),
])
def test_auto_device_selection(monkeypatch, capsys, devices, properties, expected):
    install_device_core(monkeypatch, devices, properties)
    assert generation._select_device(None) == expected
    assert "selecting " + expected in capsys.readouterr().err


def test_explicit_cpu_skips_architecture_probes(monkeypatch):
    install_device_core(monkeypatch, ["CPU", "GPU"])
    assert generation._select_device("CPU") == "CPU"


@pytest.mark.parametrize("device", ["GPU", "NPU"])
def test_explicit_unavailable_device(monkeypatch, device):
    install_device_core(monkeypatch, ["CPU"])
    with pytest.raises(RuntimeError, match=f"{device} is unavailable"):
        generation._select_device(device)


def test_auto_no_cpu_fallback_is_actionable(monkeypatch):
    install_device_core(monkeypatch, [])
    with pytest.raises(RuntimeError, match="CPU fallback is unavailable"):
        generation._select_device(None)


@pytest.mark.parametrize("stage", ["compile", "inference"])
def test_auto_failure_never_retries(monkeypatch, tmp_path, stage):
    pipe, calls = install_fakes(monkeypatch, ("CPU", "NPU.0"))
    cache = tmp_path / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    if stage == "compile":
        monkeypatch.setattr(pipe, "compile", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    else:
        monkeypatch.setattr(type(pipe), "__call__", lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("failure")))
    with pytest.raises(RuntimeError, match="failure"):
        generation.generate("x", None, None, generation.MODEL, None, 2)
    assert len(calls) == 1 and calls[0][1]["device"] == "NPU.0"
    assert pipe.calls[0][0] == "progress" or pipe.calls[0][0] == "reshape"


def test_auto_concrete_gpu_passed_to_pipeline(monkeypatch, tmp_path):
    pipe, calls = install_fakes(monkeypatch, ("CPU", "GPU.0", "GPU.1"))
    install_device_core(monkeypatch, ["CPU", "GPU.0", "GPU.1"], {
        ("GPU.0", "name"): "Intel integrated", ("GPU.0", "type"): "integrated",
        ("GPU.1", "name"): "Intel discrete", ("GPU.1", "type"): "discrete",
    })
    cache = tmp_path / "cache"
    seed_cache(pipe, cache)
    monkeypatch.setattr(generation, "_cache_path", lambda model=generation.MODEL: cache)
    generation.generate("x", None, None, generation.MODEL, None, 2)
    assert calls[0][1]["device"] == "GPU.1"
