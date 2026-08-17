#!/usr/bin/env python3
"""Turn an ai8xize-generated network into a Rust `Network` descriptor.

Reads the `cnn.c`, `weights.h` and `main.c` that `ai8xize.py` emits for a
MAX78002 target and writes a module defining one `const NETWORK`.

Register values are emitted as `from_bits(0x...)` rather than builder chains.
izer has already computed each word; decomposing it into named fields would
drop any bit the field model does not cover, so passing the word through whole
is lossless by construction. The descriptor is still typed - `Debug` decodes it
and `Network::validate()` checks it.

    cnn-gen.py <network-dir> -o net.rs [--name kws20] [--verify]

`--verify` replays the parsed model back into a poke list and compares it to
the one in `cnn.c`, byte for byte.
"""

import argparse
import re
import sys
from pathlib import Path

QUADRANT_BASE = 0x5100_0000
QUADRANT_STRIDE = 0x0100_0000
LAYER_BASE = 0x0010_0000
KERNEL_BASE = 0x0040_0000
BIAS_BASE = 0x0018_0000
DATA_BASE = 0x0080_0000
MEMORY_STRIDE = 0x0002_0000
QUADRANTS = 4

# Quadrant control block: LCNT_MAX holds the last layer in [7:0] and the first
# in [15:8]. Note SRAM control sits at 0x04, so matching that reads 0x040e.
LCNT_OFFSET = 0x08
MAX_LAYERS = 128

# Offset, Layer field, value type, and whether the field is per quadrant.
REGISTERS = [
    (0x00, "next", "Nxtlyr", False),
    (0x04, "rows", "Rcnt", False),
    (0x08, "cols", "Ccnt", False),
    (0x0C, "oned", "Oned", False),
    (0x10, "pool_rows", "Prcnt", False),
    (0x14, "pool_cols", "Pccnt", False),
    (0x18, "stride", "Stride", False),
    (0x1C, "wptr", "WptrBase", True),
    (0x20, "wptr_ts", "WptrToffs", False),
    (0x24, "wptr_moffs", "WptrMoffs", False),
    (0x28, "wptr_choffs", "WptrChoffs", False),
    (0x2C, "rptr", "RptrBase", False),
    (0x30, "lctl", "Lctl", True),
    (0x34, "lctl2", "Lctl2", False),
    (0x38, "mcnt1", "Mcnt1", False),
    (0x3C, "mcnt2", "Mcnt2", False),
    (0x40, "ochan", "Ochan", False),
    (0x44, "tptr", "Tptr", False),
    (0x48, "ena", "Ena", True),
    (0x4C, "post", "Post", True),
]

BY_OFFSET = {off: (name, ty, per_q) for off, name, ty, per_q in REGISTERS}
OPTIONAL = {0x24, 0x38}

# Must match `emit_layer` in src/cnn/config.rs.
EMIT_ORDER = [
    0x00, 0x04, 0x08, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28,
    0x2C, 0x30, 0x34, 0x38, 0x3C, 0x40, 0x0C, 0x44, 0x4C, 0x48,
]

POKE = re.compile(
    r"\*\(\(volatile uint32_t \*\)\s*0x([0-9a-f]{8})\)\s*= 0x([0-9a-f]{8});"
)


def find(root, name):
    """The named file, at `root` or anywhere below it."""
    if (root / name).exists():
        return root / name
    for path in sorted(root.rglob(name)):
        return path
    sys.exit(f"no {name} under {root}")


def section(text, signature, end="return CNN_OK;"):
    """The body of a generated function, or None if it is absent."""
    try:
        start = text.index(signature)
    except ValueError:
        return None
    return text[start : text.index(end, start)]


def parse_layers(cnn_c):
    """Layer register values, indexed by layer then quadrant then offset."""
    body = section(cnn_c, "int cnn_configure(void)")
    if body is None:
        sys.exit("cnn.c has no cnn_configure")

    layers = {}
    order = []
    for m in POKE.finditer(body):
        addr, value = int(m.group(1), 16), int(m.group(2), 16)
        quadrant = (addr >> 24) - 0x51
        inner = addr & 0x00FF_FFFF
        if not 0 <= quadrant < QUADRANTS or inner >> 20 != 1:
            sys.exit(f"unexpected configure write to {addr:#010x}")
        index, offset = (inner & 0xFFFFF) >> 8, inner & 0xFF
        if index >= MAX_LAYERS or offset not in BY_OFFSET:
            sys.exit(
                f"write to {addr:#010x} is not a layer register. Streaming and "
                "FIFO networks configure registers this HAL does not support."
            )
        layers.setdefault(index, {}).setdefault(quadrant, {})[offset] = value
        order.append((index, quadrant, offset, value))
    return layers, order


def parse_layer_range(cnn_c):
    """`first_layer` and `last_layer`, from the LCNT_MAX write in cnn_init."""
    body = section(cnn_c, "int cnn_init(void)")
    for m in POKE.finditer(body):
        addr, value = int(m.group(1), 16), int(m.group(2), 16)
        if addr & 0x00FF_FFFF == LCNT_OFFSET:
            return value >> 8 & 0xFF, value & 0xFF
    sys.exit("cnn_init has no layer count write")


def parse_blob(header, name):
    """The values of a `#define <NAME> { ... }` initializer in weights.h."""
    m = re.search(rf"^#define\s+{name}\s*\\?\s*\n?\s*\{{", header, re.M)
    if not m:
        return None
    depth, i = 0, m.end() - 1
    while i < len(header):
        depth += (header[i] == "{") - (header[i] == "}")
        if depth == 0:
            break
        i += 1
    values = header[m.end() : i].replace("\\\n", " ")
    return [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|-?\b\d+\b", values)]


def parse_weights(header):
    """Weight regions, from the self-describing `kernels` blob."""
    words = parse_blob(header, "KERNELS")
    if words is None:
        sys.exit("weights.h has no kernels blob")

    regions, i = [], 0
    while i < len(words) and words[i] != 0:
        addr, length = words[i], words[i + 1]
        data = words[i + 2 : i + 2 + length]
        i += 2 + length

        quadrant = (addr >> 24) - 0x51
        inner = addr & 0x00FF_FFFF
        processor = (inner - KERNEL_BASE) // MEMORY_STRIDE
        offset = (inner - KERNEL_BASE) % MEMORY_STRIDE
        regions.append((quadrant, processor, offset, data))
    return regions


def parse_bias(cnn_c, header):
    """Per-quadrant bias tables, or None when the network has no bias."""
    body = section(cnn_c, "int cnn_load_bias(void)")
    calls = re.findall(
        r"memcpy_8to32\(\(uint32_t \*\)\s*0x([0-9a-f]{8}),\s*(\w+),\s*"
        r"sizeof\(uint8_t\) \* (\d+)\)",
        body or "",
    )
    if not calls:
        return None

    tables = [[] for _ in range(QUADRANTS)]
    for addr, symbol, length in calls:
        quadrant = (int(addr, 16) >> 24) - 0x51
        values = parse_blob(header, symbol.upper())
        if values is None:
            sys.exit(f"weights.h has no {symbol}")
        tables[quadrant] = values[: int(length)]
    return tables


def parse_regions(text, pattern):
    """(quadrant, instance, word, len) for each data memory run."""
    regions = []
    for addr, length in pattern.findall(text):
        addr = int(addr, 16)
        quadrant = (addr >> 24) - 0x51
        inner = (addr & 0x00FF_FFFF) - DATA_BASE
        instance, byte = divmod(inner, MEMORY_STRIDE)
        regions.append((quadrant, instance, byte // 4, int(length)))
    return regions


def parse_input(main_c):
    return parse_regions(
        main_c,
        re.compile(r"memcpy32\(\(uint32_t \*\)\s*0x([0-9a-f]{8}),\s*\w+,\s*(\d+)\)"),
    )


def parse_output(cnn_c):
    """Output runs, from the unrolled pointer walk in cnn_unload."""
    body = section(cnn_c, "int cnn_unload(uint32_t *out_buf)", "return CNN_OK;")
    if body is None:
        return []

    regions, addr, count = [], None, 0
    for line in body.splitlines():
        start = re.search(r"addr = \(volatile uint32_t \*\)\s*0x([0-9a-f]{8});", line)
        if start:
            if addr is not None:
                regions.append((addr, count))
            addr, count = int(start.group(1), 16), 0
        elif "*out_buf++ = *addr++;" in line:
            count += 1
    if addr is not None:
        regions.append((addr, count))

    out = []
    for addr, count in regions:
        quadrant = (addr >> 24) - 0x51
        inner = (addr & 0x00FF_FFFF) - DATA_BASE
        instance, byte = divmod(inner, MEMORY_STRIDE)
        out.append((quadrant, instance, byte // 4, count))
    return out


def replay(layers):
    """The poke list the HAL would emit for this model, in emit order."""
    out = []
    for index in sorted(layers):
        for quadrant in sorted(layers[index]):
            values = layers[index][quadrant]
            for offset in EMIT_ORDER:
                value = values.get(offset, 0)
                if value != 0:
                    out.append((index, quadrant, offset, value))
    return out


def verify(layers, order):
    expected = [(i, q, o, v) for i, q, o, v in order]
    actual = replay(layers)
    if actual == expected:
        print(f"verify: {len(actual)} writes match")
        return 0

    print(f"verify: FAILED, {len(actual)} emitted against {len(expected)}")
    for n, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            print(f"  first difference at {n}: emitted {a}, expected {e}")
            break
    return 1


def rust_layer(index, quadrants):
    """One `Layer` literal, listing only the registers that were written."""
    shared, per_quadrant = [], {}
    for offset, name, ty, per_q in REGISTERS:
        if per_q:
            values = [quadrants.get(q, {}).get(offset, 0) for q in range(QUADRANTS)]
            if any(values):
                per_quadrant[name] = (ty, values)
            continue
        value = quadrants.get(0, {}).get(offset, 0)
        if value == 0:
            continue
        literal = f"{ty}::from_bits({value:#010x})"
        shared.append(f"        {name}: Some({literal})," if offset in OPTIONAL
                      else f"        {name}: {literal},")

    lines = [f"    // layer {index}", "    Layer {"]
    lines += shared
    for name, (ty, values) in per_quadrant.items():
        rendered = ", ".join(f"{ty}::from_bits({v:#010x})" for v in values)
        lines.append(f"        {name}: [{rendered}],")
    lines.append("        ..Layer::new()")
    lines.append("    },")
    return lines


def rust_slice(name, ty, count, lines):
    """A `const NAME: [TY; count]` whose body is already rendered."""
    return [f"const {name}: [{ty}; {count}] = ["] + lines + ["];"]


def emit(name, layers, first, last, weights, bias, inputs, outputs):
    out = [
        "// Generated by tools/cnn-gen.py. Do not edit.",
        "",
        "use max7800x_hal::cnn::fields::*;",
        "use max7800x_hal::cnn::{Direct, InputRegion, Layer, Network, "
        "OutputRegion, WeightRegion};",
        "",
    ]

    body = []
    for index in sorted(layers):
        body += rust_layer(index, layers[index])
    out += rust_slice("LAYERS", "Layer", len(layers), body) + [""]

    for n, (_, _, _, data) in enumerate(weights):
        values = ", ".join(f"{w:#010x}" for w in data)
        out.append(f"static KERNEL_{n}: [u32; {len(data)}] = [{values}];")
    out.append("")

    rows = []
    for n, (quadrant, processor, offset, data) in enumerate(weights):
        rows.append(
            f"    WeightRegion {{ quadrant: {quadrant}, processor: {processor}, "
            f"offset: {offset:#x}, data: &KERNEL_{n} }},"
        )
    out += rust_slice("WEIGHTS", "WeightRegion", len(rows), rows) + [""]

    if bias is not None:
        for quadrant, table in enumerate(bias):
            values = ", ".join(str(v) for v in table)
            out.append(f"static BIAS_{quadrant}: [u8; {len(table)}] = [{values}];")
        tables = ", ".join(f"&BIAS_{q}" for q in range(QUADRANTS))
        out.append(f"const BIAS: [&[u8]; 4] = [{tables}];")
        out.append("")

    for const, ty, regions in [
        ("INPUT", "InputRegion", inputs),
        ("OUTPUT", "OutputRegion", outputs),
    ]:
        rows = [
            f"    {ty} {{ quadrant: {q}, instance: {i}, word: {w}, len: {n} }},"
            for q, i, w, n in regions
        ]
        out += rust_slice(const, ty, len(rows), rows) + [""]

    out += [
        f"/// {name}, generated from ai8xize output",
        "pub const NETWORK: Network<'static, Direct> = Network::new(",
        "    &LAYERS,",
        f"    {first},",
        f"    {last},",
        "    &WEIGHTS,",
        "    Some(&BIAS)," if bias is not None else "    None,",
        "    &INPUT,",
        "    &OUTPUT,",
        ");",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("network", type=Path, help="directory holding cnn.c")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    source = find(args.network, "cnn.c")
    cnn_c = source.read_text()
    header = find(source.parent, "weights.h").read_text()
    main_c = "".join(p.read_text() for p in args.network.rglob("main.c"))

    layers, order = parse_layers(cnn_c)
    first, last = parse_layer_range(cnn_c)

    if args.verify and verify(layers, order):
        return 1
    if not args.output:
        return 0

    text = emit(
        args.name or args.network.name,
        layers,
        first,
        last,
        parse_weights(header),
        parse_bias(cnn_c, header),
        parse_input(main_c),
        parse_output(cnn_c),
    )
    args.output.write_text(text)
    print(f"wrote {args.output} ({len(layers)} layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
