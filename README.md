# max7800x-hal
[![Crates.io Version](https://img.shields.io/crates/v/max7800x-hal)](https://crates.io/crates/max7800x-hal)
[![docs.rs](https://img.shields.io/docsrs/max7800x-hal)](https://docs.rs/max7800x-hal)

This is an [Embedded HAL] (Hardware Abstraction Layer) for the MAX78000 and MAX78002 microcontrollers from Analog Devices.

The HAL is built on top of a Peripheral Access Crate, which provides low-level access to the microcontroller's registers. The HAL provides a higher-level interface to the peripherals, making it easier to write applications.

[Embedded HAL]: https://crates.io/crates/embedded-hal
[`max78000-pac`]: https://github.com/sigpwny/max78000-pac
[`max78002-pac`]: https://github.com/vinayak-vikram/max78002-pac

## Target selection
Exactly one target feature must be enabled. `max78000` is the default and pulls in [`max78000-pac`]; `max78002` pulls in [`max78002-pac`] instead.

```toml
max7800x-hal = { version = "0.7.1", default-features = false, features = ["max78002", "rand", "rt"] }
```

The target feature selects the PAC re-exported as `hal::pac` along with the
chip-specific constants: IPO frequency (100 MHz vs 120 MHz), flash size and
page size (512 KiB / 8 KiB vs 2.5 MiB / 16 KiB), and the pins available on
each GPIO port.

## CNN accelerator (MAX78002 only)

The `cnn` module drives the MAX78002's CNNx16 accelerator: four quadrants of
sixteen processors, up to 128 layers. It is gated on the `max78002` feature and
needs a `max78002-pac` with the CNN peripherals (`Cnn`, `Cnnx16_0` through
`Cnnx16_3`, `Gcfr`), which is why the dependency is pinned to the `main` branch
of the git repository rather than a crates.io release.

A `Network` is a `const` descriptor — layer register values, weight blobs, and
where input and output live in data memory. It goes in flash and is read by
reference. Nothing in the HAL computes a register value; that arithmetic
belongs to the network compiler.

```rust
let mut cnn = Cnn::new(p.cnn, p.cnnx16_0, p.cnnx16_1, p.cnnx16_2, p.cnnx16_3, p.gcfr)
    .enable(&mut gcr, CnnClockSource::Ipll(pll), CnnClockDiv::Div1, &mut delay);

NETWORK.validate().unwrap();
cnn.init(&NETWORK);
cnn.load_weights(&NETWORK);
cnn.load_bias(&NETWORK);
cnn.configure(&NETWORK);

cnn.write_u32(&NETWORK, &samples);
cnn.start();
cnn.wait();
cnn.read_u32(&NETWORK, &mut out);
```

### Generating a network

`tools/cnn-gen.py` turns the output of [ai8xize.py] into that descriptor:

```sh
python3 tools/cnn-gen.py path/to/generated/network -o src/network.rs
python3 tools/cnn-gen.py path/to/generated/network --verify
```

`--verify` replays the parsed model back into a register poke list and compares
it against the one in `cnn.c`, byte for byte. Register values are emitted as
`from_bits(0x...)` rather than builder chains, so no bit is lost to a field the
model does not name.

[ai8xize.py]: https://github.com/analogdevicesinc/ai8x-synthesis

### Memory

Weights and bias live in flash as `static` arrays and are copied into the
accelerator at startup, so no `memory.x` change is needed — but a large network
is most of the flash budget. Capacities per part:

| Memory | Per unit | Units | Total |
| --- | --- | --- | --- |
| Kernel | 4096 kernels, 5120 on processor 0 | 64 processors | 2,396,160 B |
| Bias | 2048 entries | 4 quadrants | 8,192 B |
| Data SRAM | 5120 words | 16 instances | 327,680 B |
| TRAM | 12288 words | 64 processors | — |

Data SRAM holds the input, every intermediate activation and the output at
once, which is the limit a large input runs into first.

### Scope

Direct input only. FIFO input and streaming layers are not implemented: the
target workload feeds buffered sample windows, which are already in RAM by the
time the accelerator needs them. `Network::validate` rejects what it can detect
of an unsupported network and `cnn-gen.py` refuses to parse one; the trait that
selected between the two is kept, commented out, at the top of `cnn::network`.

MAX78000 is out of scope — its layer register file is register-major with a
32-layer cap, a different backend rather than a variation.

### On the register spec

The layer register bitfields are **not** from a vendor document. No Analog
Devices datasheet or user guide describes them. The field model in
`cnn::fields` was reverse-engineered from `izer/tornadocnn.py` and
`izer/backend/max7800x.py` in ai8x-synthesis, then checked against the register
values in every MAX78002 CNN example shipped with the MSDK.

Bit positions are reliable; conditional logic around them is less so. The tests
in `cnn::golden` replay 4,559 real register writes through the emit path, and
`declared_bits_match_the_getters` pins each reserved-bit mask to the fields
actually declared. Treat a field name as a good hypothesis, not a guarantee.

## Roadmap
See the [roadmap] to see current implementation progress and future plans.

[roadmap]: https://github.com/sigpwny/max7800x-hal/issues/1

> [!NOTE]  
> This HAL is under active development. As a result, the API is volatile and subject to change. Be sure to refer to the [changelog] for breaking changes.

If you want updates for when new releases are made, you can watch this repository by clicking the "Watch" button at the top of the page.

[changelog]: https://github.com/sigpwny/max7800x-hal/releases

## Getting Started
If you already have an existing Rust project, you can add this crate by running:
```sh
cargo add max7800x-hal
```

Otherwise, we recommend getting started using this [Crate template for the MAX78000FTHR board](https://github.com/sigpwny/max78000fthr-template). If you are not using the MAX78000FTHR board, you can still use the template as a reference for setting up your own project.

```sh
cargo generate --git https://github.com/sigpwny/max78000fthr-template
```

## Documentation
Documentation for this HAL can be built by running:
```sh
cargo doc --open
```

Documentation can also be found on [docs.rs](https://docs.rs/max7800x-hal).

## Contributing
We welcome contributions from the community! If you want to contribute to this project, follow the steps below to get started:
1. Fork the repository to create your own copy.
2. Make your changes, commit them, then push them to your fork.
3. Open a [pull request](https://github.com/sigpwny/max7800x-hal/pulls).
4. Maintainers will review your PR and suggest changes if needed.
5. Get merged!

## Maintainers
- [SIGPwny](https://sigpwny.com) of the University of Illinois Urbana-Champaign

We are happy to invite additional maintainers to this repository, especially those who are involved in the eCTF competition! You can contact us at `hello@sigpwny.com` or via [Discord](https://sigpwny.com/discord) to request becoming a maintainer.

## License
This template is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

Copyright (c) 2025 SIGPwny
