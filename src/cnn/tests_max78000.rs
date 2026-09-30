//! MAX78000 checks. Addresses and values are cross-checked against the
//! MaximSDK MAX78000 CNN examples, `izer/tornadocnn.py` (`DevAI85`) in
//! ai8x-synthesis, and chapter 27 of the MAX78000 User Guide.

use super::config::*;
use super::fields::*;
use super::memory::*;
use super::network::*;
use super::regs::*;
use super::validate::*;
use crate::pac;

// ---------------------------------------------------------------- addresses

#[test]
fn quadrant_bases_match_pac() {
    assert_eq!(quadrant_base(0), pac::Cnnx16_0::PTR as u32);
    assert_eq!(quadrant_base(1), pac::Cnnx16_1::PTR as u32);
    assert_eq!(quadrant_base(2), pac::Cnnx16_2::PTR as u32);
    assert_eq!(quadrant_base(3), pac::Cnnx16_3::PTR as u32);
    assert_eq!(quadrant_base(3), 0x50d0_0000);
}

#[test]
fn layer_registers_are_register_major() {
    // UG7456 27.5.2.1: CNNx16_n_L21_RCNT is at offset 0x10 + 4 * 21.
    assert_eq!(
        Quadrant::new(0).layer(21).reg(LayerReg::Rows).addr(),
        0x5010_0064
    );
    // kws20_demo layer 8: write pointer, LCTRL1 and enables.
    let l8 = Quadrant::new(0).layer(8);
    assert_eq!(l8.reg(LayerReg::Wptr).addr(), 0x5010_0330);
    assert_eq!(l8.reg(LayerReg::Lctl2).addr(), 0x5010_0a30);
    assert_eq!(l8.reg(LayerReg::En).addr(), 0x5010_0730);
    // The last layer's copy stays inside the register's 0x80-byte array.
    let last = Quadrant::new(3).layer(MAX_LAYERS - 1);
    assert_eq!(last.reg(LayerReg::Post).addr(), 0x50d0_080c);
}

#[test]
fn memory_addresses() {
    // weights.h: kernel records start at 0x50180000, 0x50184000, ...
    assert_eq!(kernel_addr(0, 0), 0x5018_0000);
    assert_eq!(kernel_addr(0, 1), 0x5018_4000);
    assert_eq!(kernel_addr(3, 15), 0x50dbc000);
    // cifar-10 cnn_load_bias.
    assert_eq!(bias_addr(0), 0x5010_8000);
    assert_eq!(bias_addr(3), 0x50d0_8000);
    // UG7456 figure 27-3.
    assert_eq!(tram_addr(0, 0), 0x5011_0000);
    assert_eq!(tram_addr(0, 15), 0x5014_c000);
    // kws20_demo cnn_unload.
    assert_eq!(data_addr(0, 0) + 0x2000, 0x5040_2000);
    assert_eq!(data_addr(0, 1) + 0x2000, 0x5040_a000);
    assert_eq!(data_addr(1, 0) + 0x2000, 0x5080_2000);
}

#[test]
fn capacities_match_the_datasheet() {
    let kernels: u32 = (0..PROCESSORS_PER_QUADRANT).map(kernel_capacity).sum();
    assert_eq!(kernels * QUADRANTS as u32 * 9, TOTAL_KERNEL_BYTES);
    // 512KB of data memory, fully backed.
    assert_eq!(
        DATA_WINDOW_BYTES * DATA_INSTANCES_PER_QUADRANT as u32 * QUADRANTS as u32,
        512 * 1024
    );
    assert_eq!(DATA_INSTANCE_WORDS, DATA_INSTANCE_STRIDE_WORDS);
    // A kernel is written as four words.
    assert!(kernel_burst_fits(0, KERNELS_PER_PROCESSOR as usize * 4));
}

// ---------------------------------------------------------------- fields

#[test]
fn typed_registers_cover_every_slot_once() {
    assert_eq!(ALL_TYPED_REGS.len(), ALL_LAYER_REGS.len());
    for &expected in ALL_LAYER_REGS {
        let count = ALL_TYPED_REGS.iter().filter(|r| **r == expected).count();
        assert_eq!(count, 1, "{expected:?} has {count} value types");
    }
}

/// Same check as the MAX78002 suite: a bit is declared exactly when toggling
/// it changes what the `Debug` impl renders.
#[test]
fn declared_bits_match_the_getters() {
    for &reg in ALL_TYPED_REGS {
        let (zero, ones) = (rendered(reg, 0), rendered(reg, u32::MAX));
        for bit in 0..32 {
            let up = rendered(reg, 1 << bit);
            let down = rendered(reg, u32::MAX ^ (1 << bit));
            let visible = up.as_str() != zero.as_str() || down.as_str() != ones.as_str();
            let declared = reserved_bits(reg, 1 << bit) == 0;
            assert_eq!(visible, declared, "{reg:?} bit {bit}");
        }
    }
}

/// The reserved windows from the register tables in UG7456 27.5.2.1.
#[test]
fn reserved_windows_match_the_user_guide() {
    for (reg, bits) in [
        (LayerReg::Rows, 0xfffc_fc00u32),
        (LayerReg::Cols, 0xfffc_fc00),
        (LayerReg::Oned, 0xffc0_0000),
        (LayerReg::PoolRows, 0xffff_fff0),
        (LayerReg::PoolCols, 0xffff_fff0),
        (LayerReg::Stride, 0xffff_fffc),
        (LayerReg::Wptr, 0xfffe_0000),
        (LayerReg::WptrTs, 0xfffe_0000),
        (LayerReg::WptrMask, 0xfffe_0000),
        (LayerReg::WptrMp, 0xfffe_0000),
        (LayerReg::Rptr, 0xfffe_0000),
        (LayerReg::Lctl, 0xfffe_041f),
        (LayerReg::Mcnt, 0),
        (LayerReg::Tptr, 0xf800_f800),
        (LayerReg::En, 0),
        (LayerReg::Post, 0xe000_0000),
        (LayerReg::Lctl2, 0xfffe_0000),
    ] {
        assert_eq!(reserved_bits(reg, u32::MAX), bits, "{reg:?}");
    }
}

#[test]
fn kws20_layer0_decodes() {
    // Layer 0 quadrant 0 of kws20_demo.
    let lctl = Lctl::from_bits(0x0000_eb20);
    assert!(lctl.relu() && lctl.maxpool() && lctl.global_wptr());
    assert_eq!(lctl.siena(), 0b1110);
    let lctl2 = Lctl2::from_bits(0x0001_9811);
    assert_eq!(
        (lctl2.maxpass(), lctl2.wptr_inc(), lctl2.xpch_max()),
        (1, 1, 0x198)
    );
    let mcnt = Mcnt::from_bits(0x0240_0838);
    assert_eq!((mcnt.mcnt_sad(), mcnt.mcnt_max()), (0x240, 0x838));
    let rows = Rcnt::from_bits(0x0001_007f);
    assert_eq!((rows.cnt(), rows.pad()), (0x7f, 1));
}

/// A `core::fmt` sink, so the decoder can be checked without `std`.
struct Buf {
    data: [u8; 512],
    len: usize,
}

impl Buf {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len]).unwrap()
    }
}

impl core::fmt::Write for Buf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let n = s.len().min(self.data.len() - self.len);
        self.data[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

fn rendered(reg: LayerReg, bits: u32) -> Buf {
    use core::fmt::Write;
    let mut buf = Buf {
        data: [0; 512],
        len: 0,
    };
    with_decoded(reg, bits, |v| write!(buf, "{v:?}").unwrap());
    buf
}

// ---------------------------------------------------------------- golden corpus

/// `kws20_demo`: 9 layers, no bias, conv1d
const KWS20: &str = include_str!("golden/max78000/kws20_demo.txt");

/// `cifar-10`: 11 layers, bias
const CIFAR10: &str = include_str!("golden/max78000/cifar10.txt");

const CORPUS: [(&str, &str, usize); 2] = [("kws20_demo", KWS20, 9), ("cifar10", CIFAR10, 11)];

/// One `// Layer n quadrant q` block, as the generator emitted it
struct Block {
    quadrant: u8,
    layer: u8,
    writes: [(LayerReg, u32); 17],
    len: usize,
}

impl Block {
    fn writes(&self) -> &[(LayerReg, u32)] {
        &self.writes[..self.len]
    }
}

/// Splits a fixture into blocks, each a maximal run sharing a layer and quadrant
fn for_each_block(src: &str, mut f: impl FnMut(&Block)) {
    let mut current: Option<Block> = None;
    for line in src.lines() {
        let (addr, value) = line.split_once(' ').expect("malformed fixture line");
        let addr = u32::from_str_radix(addr, 16).unwrap();
        let value = u32::from_str_radix(value, 16).unwrap();

        let quadrant = ((addr - quadrant_base(0)) / (quadrant_base(1) - quadrant_base(0))) as u8;
        let offset = addr - quadrant_base(quadrant);
        let reg = *ALL_LAYER_REGS
            .iter()
            .find(|&&r| offset >= r as u32 && offset - (r as u32) < 4 * MAX_LAYERS as u32)
            .expect("address is a layer register");
        let layer = ((offset - reg as u32) / 4) as u8;

        if current
            .as_ref()
            .is_some_and(|b| b.quadrant != quadrant || b.layer != layer)
        {
            f(current.as_ref().unwrap());
            current = None;
        }
        let block = current.get_or_insert(Block {
            quadrant,
            layer,
            writes: [(LayerReg::Rows, 0); 17],
            len: 0,
        });
        block.writes[block.len] = (reg, value);
        block.len += 1;
    }
    if let Some(block) = current.as_ref() {
        f(block);
    }
}

/// Rebuilds the layer a block was emitted from, ignoring order
fn layer_from(block: &Block) -> Layer {
    let q = block.quadrant as usize;
    let mut l = Layer::default();
    for &(reg, v) in block.writes() {
        match reg {
            LayerReg::Rows => l.rows = Rcnt::from_bits(v),
            LayerReg::Cols => l.cols = Ccnt::from_bits(v),
            LayerReg::Oned => l.oned = Oned::from_bits(v),
            LayerReg::PoolRows => l.pool_rows = Prcnt::from_bits(v),
            LayerReg::PoolCols => l.pool_cols = Pccnt::from_bits(v),
            LayerReg::Stride => l.stride = Stride::from_bits(v),
            LayerReg::Wptr => l.wptr[q] = WptrBase::from_bits(v),
            LayerReg::WptrTs => l.wptr_ts = WptrToffs::from_bits(v),
            LayerReg::WptrMask => l.wptr_moffs = Some(WptrMoffs::from_bits(v)),
            LayerReg::WptrMp => l.wptr_choffs = WptrChoffs::from_bits(v),
            LayerReg::Rptr => l.rptr = RptrBase::from_bits(v),
            LayerReg::Lctl => l.lctl[q] = Lctl::from_bits(v),
            LayerReg::Mcnt => l.mcnt = Mcnt::from_bits(v),
            LayerReg::Tptr => l.tptr = Tptr::from_bits(v),
            LayerReg::En => l.ena[q] = Ena::from_bits(v),
            LayerReg::Post => l.post[q] = Post::from_bits(v),
            LayerReg::Lctl2 => l.lctl2 = Lctl2::from_bits(v),
        }
    }
    l
}

struct Recorder {
    writes: [(LayerReg, u32); 17],
    len: usize,
}

impl LayerSink for Recorder {
    fn write(&mut self, reg: LayerReg, bits: u32) {
        self.writes[self.len] = (reg, bits);
        self.len += 1;
    }
}

/// Every block, re-emitted from a rebuilt layer, must come back in the same
/// order with the same zero-write suppression.
#[test]
fn every_block_round_trips() {
    for (name, src, layers) in CORPUS {
        let mut blocks = 0;
        for_each_block(src, |block| {
            let mut recorder = Recorder {
                writes: [(LayerReg::Rows, 0); 17],
                len: 0,
            };
            emit_layer(&mut recorder, &layer_from(block), block.quadrant);
            assert_eq!(
                &recorder.writes[..recorder.len],
                block.writes(),
                "{name} layer {} quadrant {}",
                block.layer,
                block.quadrant
            );
            blocks += 1;
        });
        assert_eq!(blocks, layers * QUADRANTS as usize, "{name}");
    }
}

/// Only `WPTR`, `LCTRL0`, `EN` and `POST` differ between quadrants, which is
/// what `Layer` assumes.
#[test]
fn shared_registers_agree_across_quadrants() {
    let per_quadrant = [LayerReg::Wptr, LayerReg::Lctl, LayerReg::En, LayerReg::Post];
    for (name, src, _) in CORPUS {
        let mut master = [(LayerReg::Rows, 0); 17];
        let mut master_len = 0;
        for_each_block(src, |block| {
            let shared = block
                .writes()
                .iter()
                .filter(|(r, _)| !per_quadrant.contains(r));
            if block.quadrant == 0 {
                master_len = 0;
                for &w in shared {
                    master[master_len] = w;
                    master_len += 1;
                }
                return;
            }
            assert!(
                shared.eq(master[..master_len].iter()),
                "{name} layer {} quadrant {}",
                block.layer,
                block.quadrant
            );
        });
    }
}

#[test]
fn no_shipped_value_touches_a_reserved_bit() {
    for (name, src, _) in CORPUS {
        for_each_block(src, |block| {
            for &(reg, value) in block.writes() {
                assert_eq!(
                    reserved_bits(reg, value),
                    0,
                    "{name} layer {} {reg:?} = {value:#010x}",
                    block.layer
                );
            }
        });
    }
}

#[test]
fn every_layer_validates() {
    for (name, src, _) in CORPUS {
        for_each_block(src, |block| {
            let layers = [layer_from(block)];
            let net = Network::new(&layers, 0, 0, &[], None, &[], &[]);
            assert_eq!(net.validate(), Ok(()), "{name} layer {}", block.layer);
        });
    }
}

// ---------------------------------------------------------------- validation

#[test]
fn processing_must_start_at_layer_zero() {
    let layers = [Layer::new(), Layer::new()];
    let net = Network::new(&layers, 1, 1, &[], None, &[], &[]);
    assert_eq!(
        net.validate(),
        Err(Invalid::LayerRange {
            first: 1,
            last: 1,
            layers: 2
        })
    );
}

#[test]
fn at_most_32_layers() {
    let layers = [const { Layer::new() }; 33];
    let net = Network::new(&layers, 0, 0, &[], None, &[], &[]);
    assert_eq!(net.validate(), Err(Invalid::TooManyLayers { layers: 33 }));
}

#[test]
fn bias_memory_holds_512_entries() {
    let layers = [Layer::new()];
    let bias = [0u8; 513];
    let tables: [&[u8]; 4] = [&bias[..512], &[], &[], &bias];
    let net = Network::new(&layers, 0, 0, &[], Some(&tables), &[], &[]);
    assert_eq!(net.validate(), Err(Invalid::BiasOverrun { quadrant: 3 }));
}
