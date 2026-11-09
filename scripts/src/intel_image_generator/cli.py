from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import math
from pathlib import Path


def _size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)x([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise argparse.ArgumentTypeError("width and height must be between 1 and 4096")
    return width, height


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= number <= 1000:
        raise argparse.ArgumentTypeError("must be between 1 and 1000")
    return number


def _strength(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("strength must be a finite number") from exc
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("strength must be finite and greater than 0 and at most 1")
    return number


def _seed(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= number <= 2**63 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 2**63-1")
    return number


def _model(value: str) -> str:
    parts = value.split("/")
    component = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\Z")
    if len(value) > 96 or len(parts) != 2 or any(not component.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError(
            "model must be a Hugging Face repository ID in namespace/name form, not a URL or path"
        )
    if any(".." in part or "--" in part for part in parts):
        raise argparse.ArgumentTypeError("model ID components must not contain consecutive dots or hyphens")
    if any(part.endswith(".git") for part in parts):
        raise argparse.ArgumentTypeError("model repository ID must not end in .git")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intel-image-generator",
        description="Generate an image with a compatible Diffusers Stable Diffusion text-to-image model.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-p", "--prompt", help="one text prompt")
    source.add_argument("-i", "--input", type=Path, help="UTF-8 file with one prompt per nonblank line")
    parser.add_argument("-o", "--output", help="PNG destination (required for generation)")
    parser.add_argument("-r", "--reference", type=Path, help="starting image for img2img; not an identity/style guarantee")
    parser.add_argument("-s", "--style", type=Path, help="image reference conditioning for text-to-image (not img2img)")
    parser.add_argument("--strength", type=_strength, default=None, help="img2img denoising strength (0 < value <= 1; default 0.75)")
    parser.add_argument("--info", action="store_true", help="print selected device/model JSON without generation")
    parser.add_argument("--size", type=_size, default=(512, 512), metavar="WIDTHxHEIGHT")
    parser.add_argument("--steps", type=_positive, default=20)
    parser.add_argument("--seed", type=_seed)
    parser.add_argument(
        "--device", choices=("NPU", "GPU", "CPU"), default=None,
        help=("explicit device; auto prefers a discrete Intel GPU when present, otherwise "
              "NPU then integrated Intel GPU, with CPU last; explicit choices are strict"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-m", "--model", type=_model, help="Hugging Face model (default depends on device)")
    selection.add_argument("--model-dir", type=Path, help="existing local OpenVINO export")
    return parser


def _save_image(image, destination: Path, size: tuple[int, int]) -> None:
    from io import BytesIO
    from PIL import Image, ImageOps

    fitted = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)).convert("RGB")
    encoded = BytesIO()
    fitted.save(encoded, format="PNG")
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".png", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_bytes(encoded.getvalue())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.model_dir is not None and not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")

    if args.info:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from .generation import get_info
                info = get_info(args.device, args.model_dir, args.model)
            print(json.dumps(info))
            return 0
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"intel-image-generator: {exc}", file=sys.stderr)
            return 1

    if args.style is not None and args.reference is not None:
        parser.error("--style cannot be combined with --reference")
    if args.style is not None and args.model != "stable-diffusion-v1-5/stable-diffusion-v1-5":
        parser.error("--style requires --model stable-diffusion-v1-5/stable-diffusion-v1-5; NPU int8 and custom models are unsupported")
    if args.style is not None and args.model_dir is not None:
        parser.error("--style cannot be combined with --model-dir")
    if args.strength is not None and args.reference is None:
        parser.error("--strength requires --reference")
    strength = 0.75 if args.strength is None else args.strength
    if args.prompt is None and args.input is None:
        parser.error("exactly one of -p/--prompt or -i/--input is required")
    if args.prompt is not None:
        prompts = [args.prompt] if args.prompt.strip() else []
    else:
        if not args.input.is_file():
            parser.error(f"input must be a readable regular file: {args.input}")
        try:
            prompts = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read UTF-8 input file {args.input}: {exc}")
    if not prompts:
        parser.error("prompt must not be blank" if args.prompt is not None else f"input file contains no prompts: {args.input}")
    if args.output is None:
        parser.error("the following arguments are required: -o/--output")
    if args.output == "-":
        parser.error("'-' is not supported as an output path")

    base = Path(args.output)
    destinations = ([base] if len(prompts) == 1 else [base.with_name(f"{base.stem}-{i}{base.suffix}") for i in range(1, len(prompts) + 1)])
    for destination in destinations:
        if not destination.parent.is_dir():
            parser.error(f"output parent directory does not exist: {destination.parent}")
        if destination.exists() and not destination.is_file():
            parser.error(f"output destination is not a regular file: {destination}")
    if args.input is not None and any(path.resolve() == args.input.resolve() for path in destinations):
        parser.error("an output destination aliases the input prompt file")

    reference = None
    if args.reference is not None:
        if not args.reference.is_file():
            parser.error(f"reference must be a readable regular image file: {args.reference}")
        try:
            from PIL import Image, ImageOps
            with Image.open(args.reference) as opened:
                opened.load()
                reference = ImageOps.fit(ImageOps.exif_transpose(opened).convert("RGB"), (512, 512), method=Image.Resampling.LANCZOS).copy()
        except (OSError, ValueError) as exc:
            parser.error(f"cannot decode reference image {args.reference}: {exc}")
        if any((path.exists() and os.path.samefile(path, args.reference)) or path.resolve() == args.reference.resolve() for path in destinations):
            parser.error("an output destination aliases the reference image")
    if reference is not None and int(args.steps * strength) < 1:
        parser.error("steps multiplied by img2img strength must be at least 1")

    style = None
    if args.style is not None:
        if not args.style.is_file():
            parser.error(f"style image must be a readable regular image file: {args.style}")
        try:
            from PIL import Image, ImageOps
            with Image.open(args.style) as opened:
                opened.load()
                style = ImageOps.exif_transpose(opened).convert("RGB").copy()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            parser.error(f"cannot decode style image {args.style}: {exc}")
        if any((path.exists() and os.path.samefile(path, args.style)) or path.resolve() == args.style.resolve() for path in destinations):
            parser.error("an output destination aliases the style image")

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .generation import generate_prompts
            if style is not None:
                images = generate_prompts(prompts, args.device, args.model_dir, args.model, args.seed, args.steps, style=style)
            elif reference is None:
                images = generate_prompts(prompts, args.device, args.model_dir, args.model, args.seed, args.steps)
            else:
                images = generate_prompts(prompts, args.device, args.model_dir, args.model, args.seed, args.steps, reference=reference, strength=strength)
            for image, destination in zip(images, destinations, strict=True):
                _save_image(image, destination, args.size)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"intel-image-generator: {exc}", file=sys.stderr)
        return 1
    return 0
