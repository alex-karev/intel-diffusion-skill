import builtins
import sys
from pathlib import Path

import pytest
from PIL import Image

from intel_image_generator import cli


def _args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["intel-image-generator", *map(str, args)])


def backend(monkeypatch, callback=None):
    def generate(prompts, *args):
        for prompt in prompts:
            yield callback(prompt, *args) if callback else Image.new("RGBA", (32, 16), (200, 10, 20, 30))
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", generate)


@pytest.mark.parametrize("value", ["0x1", "1x0", "-1x3", "1x-3", "4097x2", "2x4097", "1X2", "1x2x3", "1.5x2", " x2"])
def test_invalid_size(value, tmp_path, monkeypatch):
    _args(monkeypatch, "--size", value, "-o", tmp_path / "x", "-p", "p")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("value,expected", [("1x1", (1, 1)), ("4096x4096", (4096, 4096))])
def test_size_boundaries(value, expected):
    assert cli._size(value) == expected


@pytest.mark.parametrize("flag,value", [("--seed", "-1"), ("--seed", str(2**63)), ("--seed", "x"), ("--steps", "0"), ("--steps", "1001"), ("--steps", "x")])
def test_numeric_bounds_rejected(flag, value, tmp_path, monkeypatch):
    _args(monkeypatch, flag, value, "-o", tmp_path / "x", "-p", "prompt")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_numeric_boundaries():
    assert cli._seed("0") == 0
    assert cli._seed(str(2**63 - 1)) == 2**63 - 1
    assert cli._positive("1") == 1
    assert cli._positive("1000") == 1000


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_rejected(prompt, tmp_path, monkeypatch):
    _args(monkeypatch, "-o", tmp_path / "x", "-p", prompt)
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("alias", ["-o", "--output"])
def test_output_aliases(alias, tmp_path, monkeypatch):
    seen = []
    backend(monkeypatch, lambda p, *args: (seen.append(p) or Image.new("RGB", (2, 2))))
    _args(monkeypatch, alias, tmp_path / "out.png", "-p", "hello")
    assert cli.main() == 0
    assert seen == ["hello"]


def test_style_aliases_and_propagation(tmp_path, monkeypatch):
    style = tmp_path / "style.png"
    Image.new("RGBA", (4, 3), (20, 40, 60, 70)).save(style)
    calls = []
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", lambda prompts, *args, **kwargs: (calls.append((prompts, args, kwargs)) or [Image.new("RGB", (2, 2))]))
    for option in ("-s", "--style"):
        output = tmp_path / f"out-{option[1:]}.png"
        _args(monkeypatch, option, style, "--model", "stable-diffusion-v1-5/stable-diffusion-v1-5", "-o", output, "-p", "prompt")
        assert cli.main() == 0
    assert len(calls) == 2
    assert all(call[2]["style"].mode == "RGB" and call[2]["style"].size == (4, 3) for call in calls)
    assert all(call[2]["style"].getpixel((0, 0)) == (20, 40, 60) for call in calls)


@pytest.mark.parametrize("extra", [["-r", "ref.png"], ["--strength", "0.5"], ["--model", "org/custom"], ["--model-dir", "."]])
def test_style_incompatible_modes_rejected_before_backend(extra, tmp_path, monkeypatch):
    style = tmp_path / "style.png"
    Image.new("RGB", (2, 2)).save(style)
    called = []
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", lambda *a, **k: called.append((a, k)))
    _args(monkeypatch, "--style", style, "--model", "stable-diffusion-v1-5/stable-diffusion-v1-5", *extra, "-o", tmp_path / "out.png", "-p", "prompt")
    with pytest.raises(SystemExit): cli.main()
    assert not called


def test_corrupt_style_image_rejected_before_backend(tmp_path, monkeypatch):
    style = tmp_path / "broken.png"
    style.write_bytes(b"not an image")
    called = []
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", lambda *a, **k: called.append((a, k)))
    _args(monkeypatch, "--style", style, "--model", "stable-diffusion-v1-5/stable-diffusion-v1-5", "-o", tmp_path / "out.png", "-p", "prompt")
    with pytest.raises(SystemExit): cli.main()
    assert not called


def test_style_output_cannot_alias_source(tmp_path, monkeypatch):
    style = tmp_path / "style.png"
    Image.new("RGB", (2, 2)).save(style)
    called = []
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", lambda *a, **k: called.append((a, k)))
    _args(monkeypatch, "--style", style, "--model", "stable-diffusion-v1-5/stable-diffusion-v1-5", "-o", style, "-p", "prompt")
    with pytest.raises(SystemExit): cli.main()
    assert not called


@pytest.mark.parametrize("alias", ["-m", "--model"])
def test_model_aliases_and_propagation(alias, tmp_path, monkeypatch):
    seen = []
    backend(monkeypatch, lambda p, *args: (seen.append(args[2]) or Image.new("RGB", (2, 2))))
    _args(monkeypatch, alias, "org/repo", "-o", tmp_path / "out.png", "-p", "p")
    assert cli.main() == 0
    assert seen == ["org/repo"]


@pytest.mark.parametrize("model", ["org/repo", "a/b", "user_name/model.v1", "a" * 47 + "/" + "b" * 48])
def test_valid_model_ids(model):
    assert cli._model(model) == model


@pytest.mark.parametrize("model", ["", "foo", "foo/bar/baz", "https://huggingface.co/a/b", "/tmp/model", "foo/bar..baz", "foo/bar--baz", "foo/bar.", "foo/.repo", "foo/repo.git", "a" * 97 + "/b"])
def test_invalid_model_ids(model):
    with pytest.raises(Exception):
        cli._model(model)


@pytest.mark.parametrize("args", [["--model", "org/repo", "--model-dir", "/tmp", "-o", "x", "-p", "p"], ["--model-dir", "/missing", "-o", "x", "-p", "p"]])
def test_model_conflict_or_missing_directory(args, monkeypatch):
    _args(monkeypatch, *args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("device", ["NPU", "GPU", "CPU"])
def test_device_propagated(device, tmp_path, monkeypatch):
    seen = []
    backend(monkeypatch, lambda p, d, *args: (seen.append(d) or Image.new("RGB", (2, 2))))
    _args(monkeypatch, "--device", device, "-o", tmp_path / "out.png", "-p", "p")
    assert cli.main() == 0
    assert seen == [device]


def test_single_prompt_propagation_and_atomic_rgb_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "out.png"
    seen = []
    backend(monkeypatch, lambda p, *args: (seen.append((p, args)) or Image.new("RGBA", (12, 12))))
    _args(monkeypatch, "--size", "4x8", "--seed", "8", "--steps", "3", "-o", output, "-p", "hello")
    assert cli.main() == 0
    assert seen[0][0] == "hello" and seen[0][1][-2:] == (8, 3)
    with Image.open(output) as image:
        assert image.size == (4, 8) and image.mode == "RGB"
    assert capsys.readouterr().out == ""


def test_input_sequential_numbered_outputs_atomicity_and_stdout(tmp_path, monkeypatch, capsys):
    source = tmp_path / "prompts.txt"
    source.write_text(" one \n\ntwo\nthree\n")
    calls = []
    backend(monkeypatch, lambda prompt, *args: (calls.append(prompt) or Image.new("RGB", (2, 2))))
    _args(monkeypatch, "-i", source, "-o", tmp_path / "art.png", "--size", "2x2")
    assert cli.main() == 0
    assert calls == ["one", "two", "three"]
    assert [p.name for p in sorted(tmp_path.glob("art-*.png"))] == ["art-1.png", "art-2.png", "art-3.png"]
    assert source.read_text().startswith(" one")
    assert capsys.readouterr().out == ""


def test_input_alias_and_all_destinations_prevalidated(tmp_path, monkeypatch):
    source = tmp_path / "prompts.txt"
    source.write_text("one\ntwo\n")
    (tmp_path / "out-2.png").mkdir()
    called = []
    backend(monkeypatch, lambda *args: called.append(args))
    _args(monkeypatch, "-i", source, "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()
    assert called == []
    link = tmp_path / "out-1.txt"
    link.symlink_to(source)
    _args(monkeypatch, "-i", source, "-o", tmp_path / "out.txt")
    with pytest.raises(SystemExit):
        cli.main()
    assert source.read_text() == "one\ntwo\n"


@pytest.mark.parametrize("data", [b"\xff", b" \n\n"])
def test_invalid_input_rejected(tmp_path, monkeypatch, data):
    source = tmp_path / "p.txt"
    source.write_bytes(data)
    _args(monkeypatch, "-i", source, "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()


def test_missing_input_and_prompt_input_conflict_rejected(tmp_path, monkeypatch):
    _args(monkeypatch, "-i", tmp_path / "missing", "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()
    source = tmp_path / "p.txt"
    source.write_text("p")
    _args(monkeypatch, "-i", source, "-p", "p", "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()


def test_input_directory_rejected(tmp_path, monkeypatch):
    _args(monkeypatch, "-i", tmp_path, "-o", tmp_path / "out.png")
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("args", [["-o", "x"], ["-p", "p"], ["-o", "-", "-p", "p"], ["-b", "4", "-o", "x", "-p", "p"], ["bare-prompt"]])
def test_missing_output_and_removed_positional_or_batch_interface(args, monkeypatch):
    _args(monkeypatch, *args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_invalid_destinations_rejected_before_backend(tmp_path, monkeypatch):
    called = []
    backend(monkeypatch, lambda *args: called.append(args))
    for output in (tmp_path / "missing" / "x.png", tmp_path):
        _args(monkeypatch, "-o", output, "-p", "prompt")
        with pytest.raises(SystemExit):
            cli.main()
    assert called == []


def test_info_long_only_and_prompt_files_still_supported(monkeypatch, capsys):
    from intel_image_generator import generation
    monkeypatch.setattr(generation, "get_info", lambda *args: {"device": "NPU", "model": "m"})
    _args(monkeypatch, "--info")
    assert cli.main() == 0
    assert capsys.readouterr().out == '{"device": "NPU", "model": "m"}\n'
    _args(monkeypatch, "-i", "some-file", "-p", "prompt")
    with pytest.raises(SystemExit):
        cli.main()


def test_help_does_not_import_heavy_runtime(monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith(("optimum", "openvino")):
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    _args(monkeypatch, "--help")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0


@pytest.mark.parametrize("existing", [False, True])
def test_generation_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch, capsys, existing):
    output = tmp_path / "out.png"
    if existing:
        output.write_bytes(b"original")
    def fail(_prompts, *_args):
        raise RuntimeError("generation failed")
        yield
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", fail)
    _args(monkeypatch, "-o", output, "-p", "p")
    assert cli.main() == 1
    assert output.read_bytes() == b"original" if existing else not output.exists()
    assert list(tmp_path.iterdir()) == ([output] if existing else [])
    assert "generation failed" in capsys.readouterr().err


def test_midrun_failure_keeps_completed_image_and_protects_next_destination(tmp_path, monkeypatch):
    first, second = tmp_path / "out-1.png", tmp_path / "out-2.png"
    second.write_bytes(b"keep")
    def generate(prompts, *_args):
        yield Image.new("RGB", (2, 2))
        raise RuntimeError("second prompt failed")
    monkeypatch.setattr("intel_image_generator.generation.generate_prompts", generate)
    source = tmp_path / "prompts.txt"
    source.write_text("one\ntwo\n")
    _args(monkeypatch, "-i", source, "-o", tmp_path / "out.png", "--size", "2x2")
    assert cli.main() == 1
    assert first.is_file() and second.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_write_or_replace_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch, capsys, failure):
    output = tmp_path / "out.png"
    output.write_bytes(b"original")
    backend(monkeypatch)
    if failure == "replace":
        monkeypatch.setattr(cli.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace failed")))
    else:
        real_write = Path.write_bytes
        def fail_write(path, data):
            if path != output:
                raise OSError("write failed")
            return real_write(path, data)
        monkeypatch.setattr(Path, "write_bytes", fail_write)
    _args(monkeypatch, "-o", output, "-p", "p")
    assert cli.main() == 1
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]
    assert capsys.readouterr().out == ""


def test_dotted_basename_numbering(tmp_path, monkeypatch):
    backend(monkeypatch)
    source = tmp_path / "prompts.txt"
    source.write_text("one\ntwo\n")
    output = tmp_path / "my.pic.png"
    _args(monkeypatch, "-i", source, "-o", output)
    assert cli.main() == 0
    assert (tmp_path / "my.pic-1.png").is_file()
    assert (tmp_path / "my.pic-2.png").is_file()
