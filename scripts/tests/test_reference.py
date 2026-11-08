import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from intel_image_generator import cli, generation


def argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["intel-image-generator", *map(str, args)])


def install_backend(monkeypatch, callback):
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", callback)


def test_reference_preprocessed_and_strength_shared_for_prompt_file(tmp_path, monkeypatch):
    ref = tmp_path / "reference.jpg"
    Image.new("RGB", (80, 40), (230, 20, 10)).save(ref)
    original = ref.read_bytes()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one\ntwo\n")
    seen = []

    def backend(items, *args, reference=None, strength=None):
        seen.append((list(items), reference.copy(), strength))
        return iter([Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))])

    install_backend(monkeypatch, backend)
    argv(monkeypatch, "-i", prompts, "-r", ref, "-o", tmp_path / "out.png")
    assert cli.main() == 0
    assert seen[0][0] == ["one", "two"]
    assert seen[0][1].mode == "RGB" and seen[0][1].size == (512, 512)
    assert seen[0][2] == 0.75
    assert ref.read_bytes() == original


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-0.1", "1.01", "wat"])
def test_strength_invalid_values(bad, tmp_path, monkeypatch):
    argv(monkeypatch, "-p", "x", "-r", tmp_path / "missing", "--strength", bad, "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()


def test_strength_without_reference_and_zero_effective_steps(tmp_path, monkeypatch):
    argv(monkeypatch, "-p", "x", "--strength", ".5", "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()
    ref = tmp_path / "r.png"
    Image.new("RGB", (8, 8)).save(ref)
    argv(monkeypatch, "-p", "x", "-r", ref, "--steps", "1", "--strength", ".5", "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()


def test_reference_missing_corrupt_alias_rejected_before_backend(tmp_path, monkeypatch):
    called = []
    install_backend(monkeypatch, lambda *a, **kw: called.append((a, kw)))
    for ref in (tmp_path / "missing.png", tmp_path):
        argv(monkeypatch, "-p", "x", "-r", ref, "-o", tmp_path / "out.png")
        with pytest.raises(SystemExit):
            cli.main()
    corrupt = tmp_path / "bad.png"
    corrupt.write_text("not an image")
    argv(monkeypatch, "-p", "x", "-r", corrupt, "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()
    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    alias = tmp_path / "alias.png"
    alias.symlink_to(good)
    argv(monkeypatch, "-p", "x", "-r", good, "-o", alias)
    with pytest.raises(SystemExit):
        cli.main()
    assert called == []


def _backend_fakes(monkeypatch):
    pipe = types.SimpleNamespace(
        calls=[], ov_config={}, components={"vae_encoder": types.SimpleNamespace(ov_config={})}
    )
    pipe.set_progress_bar_config = lambda **kw: None
    pipe.reshape = lambda **kw: pipe.calls.append(("reshape", kw))
    pipe.compile = lambda: pipe.calls.append(("compile",))
    def invoke(**kw):
        pipe.calls.append(("call", kw))
        return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))])
    pipe.__call__ = invoke
    # Special methods resolve on class, so give a small regular fake class.
    class Pipe:
        calls = pipe.calls
        ov_config = pipe.ov_config
        components = pipe.components
        def set_progress_bar_config(self, **kw): pass
        def reshape(self, **kw): self.calls.append(("reshape", kw))
        def compile(self): self.calls.append(("compile",))
        def __call__(self, **kw):
            self.calls.append(("call", kw))
            return types.SimpleNamespace(images=[Image.new("RGB", (2, 2))])
    t2i, i2i = Pipe(), Pipe()
    calls = {"t2i": [], "i2i": []}
    class Text:
        @classmethod
        def from_pretrained(cls, source, **kw):
            calls["t2i"].append((source, kw)); return t2i
    class Img:
        @classmethod
        def from_pretrained(cls, source, **kw):
            calls["i2i"].append((source, kw)); return i2i
    module = types.ModuleType("optimum.intel.openvino")
    module.OVStableDiffusionPipeline = Text
    module.OVStableDiffusionImg2ImgPipeline = Img
    for name in ("optimum", "optimum.intel"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", module)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=lambda: types.SimpleNamespace(manual_seed=lambda s: ("gen", s))))
    class Core:
        available_devices = ["CPU"]
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    monkeypatch.setattr(generation, "_configure_compiled_cache", lambda *args: None)
    monkeypatch.setattr(generation, "_prepare_hub_pipeline", lambda *args, **kwargs: i2i)
    return t2i, i2i, calls


@pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
def test_img2img_uses_img_pipeline_reuses_generator_and_reference_copy(monkeypatch, tmp_path, device):
    t2i, i2i, calls = _backend_fakes(monkeypatch)
    monkeypatch.setattr(generation, "_select_device", lambda _: device)
    ref = Image.new("RGB", (512, 512), "red")
    results = list(generation.generate_prompts(["a", "b", "c"], device, None, "org/model", 7, 4, ref, .5))
    assert len(results) == 3
    assert calls["t2i"] == [] and calls["i2i"] == []
    assert generation._pipeline_class(True).__name__ == "Img"
    invoke_calls = [item for item in i2i.calls if item[0] == "call"]
    assert len(invoke_calls) == 3
    assert all(c[1]["strength"] == .5 and "height" not in c[1] and "width" not in c[1] for c in invoke_calls)
    assert all(c[1]["image"] is not ref for c in invoke_calls)
    assert len({id(c[1]["generator"]) for c in invoke_calls}) == 1
    assert sum(c[0] == "compile" for c in i2i.calls) == 1
    if device == "NPU":
        assert sum(c[0] == "reshape" for c in i2i.calls) == 1


def test_img2img_local_missing_encoder_errors_before_load(monkeypatch, tmp_path):
    _, _, calls = _backend_fakes(monkeypatch)
    monkeypatch.setattr(generation, "_select_device", lambda _: "CPU")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text('{"_class_name":"StableDiffusionPipeline"}')
    for comp in ("unet", "text_encoder", "vae_decoder"):
        d = model / comp; d.mkdir()
        for ext in ("xml", "bin"): (d / f"openvino_model.{ext}").touch()
    for comp, name in (("tokenizer", "tokenizer_config.json"), ("scheduler", "scheduler_config.json")):
        d = model / comp; d.mkdir(); (d / name).touch()
    with pytest.raises(RuntimeError, match="vae_encoder"):
        list(generation.generate_prompts(["x"], "CPU", model, None, None, 2, Image.new("RGB", (512, 512))))
    assert calls["t2i"] == [] and calls["i2i"] == []
