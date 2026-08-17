#!/usr/bin/env python3
"""Turn an ai8xize network configuration into a Rust `Network` descriptor.

Takes the arguments `ai8xize.py` takes - a YAML configuration, a quantized
checkpoint, a prefix - runs the real generator into a scratch directory, and
turns the C it emits into the Rust module `cnn-gen.py` would have produced.
The register values are izer's own; nothing here computes one.

    cnn-synth.py -o net.rs --config-file networks/ai87-kws20-v3-hwc.yaml \\
        --checkpoint-file trained/ai87-kws20_v3-qat8-q.pth.tar --prefix kws20

Every unrecognized argument is passed to `ai8xize.py` untouched, so anything
that changes what it emits - `--start-layer`, `--mlator`, `--no-unload`,
`--boost` - works as it does there. `--device` defaults to MAX78002 and no
other device is accepted; the HAL is ai87 only. `--test-dir` defaults to a
temporary directory that is removed on the way out, and `--no-version-check`
is added so a run needs no network.

ai8xize.py runs from its own directory, so paths are resolved against the
caller's directory first and left alone when that finds nothing - a
repository-relative `networks/x.yaml` still means what it does in the demo
scripts.

The generator lives wherever `--synthesis` or `$AI8X_SYNTHESIS` says. It runs
under `uv run --project tools`, so izer's dependencies come from this
directory's own `pyproject.toml` and `uv.lock` and no environment has to be set
up first. The pins are not decoration: numpy 2 makes izer drop the kernel
table, and torch 2.6 refuses to load the checkpoints. Pass `--python` or set
`$AI8X_PYTHON` to use an interpreter you have already prepared instead.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ai8xize.py resolves these against its own directory; the caller means theirs.
PATH_OPTIONS = ("--config-file", "--checkpoint-file", "--sample-input", "--test-dir")

SEARCH = ("~/ai8x-synthesis", "~/tmp/ai8x-synthesis")


def generator():
    """cnn-gen.py, whose file name is not an importable one."""
    path = Path(__file__).with_name("cnn-gen.py")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("cnn_gen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthesis(given):
    """The ai8x-synthesis checkout holding ai8xize.py."""
    candidates = [given] if given else [os.environ.get("AI8X_SYNTHESIS"), *SEARCH]
    for candidate in filter(None, candidates):
        root = Path(candidate).expanduser()
        if (root / "ai8xize.py").exists():
            return root.resolve()
    sys.exit("no ai8xize.py found; pass --synthesis or set $AI8X_SYNTHESIS")


def option(args, name):
    """The value of `--name value` or `--name=value`, or None."""
    for n, arg in enumerate(args):
        if arg == name and n + 1 < len(args):
            return args[n + 1]
        if arg.startswith(name + "="):
            return arg[len(name) + 1 :]
    return None


def resolve(name, value):
    """An absolute path, when the caller's directory is the one that has it."""
    path = Path(value).expanduser()
    if name == "--test-dir" or path.exists():
        return str(path.absolute())
    return value


def absolutize(args):
    """The argument list with every path option resolved."""
    out, skip = [], False
    for n, arg in enumerate(args):
        if skip:
            skip = False
            continue
        name, sep, value = arg.partition("=")
        if sep and name in PATH_OPTIONS:
            out.append(f"{name}={resolve(name, value)}")
        elif arg in PATH_OPTIONS and n + 1 < len(args):
            out += [arg, resolve(arg, args[n + 1])]
            skip = True
        else:
            out.append(arg)
    return out


def defaults(args, test_dir, prefix):
    """The argument list with the ones this tool insists on filled in."""
    args = list(args)
    device = option(args, "--device")
    if device is None:
        args += ["--device", "MAX78002"]
    elif device.upper().removeprefix("MAX") not in ("78002", "AI87", "87"):
        sys.exit(f"--device {device} is not supported; this HAL is MAX78002 only")
    if option(args, "--test-dir") is None:
        args += ["--test-dir", str(test_dir)]
    if option(args, "--prefix") is None:
        args += ["--prefix", prefix]
    if "--no-version-check" not in args:
        args.append("--no-version-check")
    return args


def interpreter(python):
    """The command that runs a script under izer's pinned dependencies."""
    if python:
        return [python]
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv not found; install it or pass --python")
    # --project, not --directory: the environment comes from here, but the
    # script still has to run from the ai8x-synthesis checkout.
    return [uv, "run", "--quiet", "--project", str(Path(__file__).parent)]


def run(root, python, args):
    """ai8xize.py, from its own directory so its assets resolve."""
    command = [*interpreter(python), "ai8xize.py", *args]
    print("+ " + " ".join(command), file=sys.stderr)
    # izer logs its progress to stdout, which is where the Rust may be going.
    result = subprocess.run(command, cwd=root, check=False, stdout=sys.stderr)
    if result.returncode:
        sys.exit(f"ai8xize.py failed ({result.returncode})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ap.add_argument("-o", "--output", type=Path, help="write Rust here, or stdout")
    ap.add_argument("--name", default=None, help="network name in the doc comment")
    ap.add_argument("--synthesis", default=None, help="ai8x-synthesis checkout")
    ap.add_argument("--python", default=None, help="interpreter to use instead of uv")
    ap.add_argument("--keep", type=Path, default=None, help="keep the C output here")
    args, passthrough = ap.parse_known_args()

    root = synthesis(args.synthesis)
    python = args.python or os.environ.get("AI8X_PYTHON")

    scratch = Path(tempfile.mkdtemp(prefix="cnn-synth-"))
    try:
        prefix = args.name or (args.output.stem if args.output else "network")
        izer_args = defaults(absolutize(passthrough), scratch / "gen", prefix)
        run(root, python, izer_args)

        # izer names the directory after the prefix; a shared --test-dir holds
        # other networks that must not be picked up instead.
        test_dir = Path(option(izer_args, "--test-dir"))
        network = test_dir / option(izer_args, "--prefix")
        if not network.is_dir():
            network = test_dir

        cnn_gen = generator()
        model = cnn_gen.parse_network(network)
        if cnn_gen.verify(model):
            return 1

        name = args.name or option(izer_args, "--prefix")
        text = cnn_gen.emit(name, model, tool=Path(__file__).name)
        if args.keep:
            shutil.copytree(network, args.keep, dirs_exist_ok=True)
        if not args.output:
            print(text, end="")
            return 0
        args.output.write_text(text)
        print(f"wrote {args.output} ({len(model.layers)} layers)")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
