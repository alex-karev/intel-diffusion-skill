import builtins
import json
import sys

import pytest

from intel_image_generator import cli, generation


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["intel-image-generator", *map(str, args)])
    return cli.main()


def forbid_model_work(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "optimum", "huggingface_hub"}:
            raise AssertionError(f"info must not import model dependencies: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        generation, "generate",
        lambda *args: pytest.fail("info must not generate an image"),
    )


@pytest.mark.parametrize("alias", ["--info"])
def test_info_uses_auto_selection_without_prompt_output_or_model_work(alias, monkeypatch, capsys):
    seen = []

    def select(requested):
        seen.append(requested)
        print("runtime diagnostic")
        return "NPU"

    monkeypatch.setattr(generation, "_select_device", select)
    forbid_model_work(monkeypatch)
    assert invoke(monkeypatch, alias) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"device": "NPU", "model": generation._default_model("NPU")}
    assert "runtime diagnostic" in captured.err
    assert seen == [None]


@pytest.mark.parametrize("alias", ["-m", "--model"])
def test_info_reports_custom_model_and_explicit_device(alias, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(generation, "_select_device", lambda device: seen.append(device) or device)
    forbid_model_work(monkeypatch)
    assert invoke(monkeypatch, "--info", "--device", "CPU", alias, "org/custom") == 0
    assert json.loads(capsys.readouterr().out) == {"device": "CPU", "model": "org/custom"}
    assert seen == ["CPU"]


def test_info_reports_resolved_local_model_path(tmp_path, monkeypatch, capsys):
    model = tmp_path / "export"
    model.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generation, "_select_device", lambda _: "GPU.1")
    forbid_model_work(monkeypatch)
    assert invoke(monkeypatch, "--info", "--model-dir", "export") == 0
    assert json.loads(capsys.readouterr().out) == {"device": "GPU.1", "model": str(model)}


def test_info_never_overwrites_output_even_when_prompt_and_output_supplied(tmp_path, monkeypatch):
    output = tmp_path / "existing.png"
    output.write_bytes(b"original")
    monkeypatch.setattr(generation, "_select_device", lambda _: "CPU")
    forbid_model_work(monkeypatch)
    assert invoke(monkeypatch, "--info", "-o", output, "-p", "ignored prompt") == 0
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]


def test_info_device_error_is_stderr_with_empty_stdout(monkeypatch, capsys):
    def unavailable(_):
        raise RuntimeError("OpenVINO device NPU is unavailable")

    monkeypatch.setattr(generation, "_select_device", unavailable)
    assert invoke(monkeypatch, "--info", "--device", "NPU") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "NPU is unavailable" in captured.err


@pytest.mark.parametrize("args", [
    ["--info", "--model", "https://huggingface.co/org/model"],
    ["--info", "--model", "org/model", "--model-dir", "."],
    ["--info", "--device", "invalid"],
])
def test_info_rejects_invalid_selection_before_runtime(args, monkeypatch):
    monkeypatch.setattr(generation, "_select_device", lambda _: pytest.fail("runtime called"))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, *args)
    assert exc.value.code == 2


def test_info_rejects_missing_local_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "_select_device", lambda _: pytest.fail("runtime called"))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "--info", "--model-dir", tmp_path / "missing")
    assert exc.value.code == 2


@pytest.mark.parametrize("args", [[], ["-o", "unused.png"]])
def test_generation_still_requires_prompt_and_output(args, monkeypatch):
    monkeypatch.setattr(generation, "generate", lambda *args: pytest.fail("generation called"))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, *args)
    assert exc.value.code == 2
