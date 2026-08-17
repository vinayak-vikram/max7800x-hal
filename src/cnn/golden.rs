//! Emit-path tests against the poke lists in generated cnn.c

use super::config::{emit_layer, LayerSink};
use super::fields::*;
use super::network::{Direct, Layer, Network};
use super::regs::{LayerReg, ALL_LAYER_REGS, QUADRANTS};

/// `kws20_demo`: 9 layers, no bias, conv1d
const KWS20: &str = include_str!("golden/kws20_demo.txt");

/// `cifar-100-mobilenet-v2-0.75`: 73 layers, bias
const CIFAR100: &str = include_str!("golden/cifar100_mobilenet_v2.txt");

/// One `// Layer n quadrant q` block, as the generator emitted it
#[derive(Clone, Copy)]
struct Block {
    quadrant: u8,
    layer: u8,
    writes: [(u8, u32); 20],
    len: usize,
}

impl Block {
    fn writes(&self) -> &[(u8, u32)] {
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

        let quadrant = ((addr >> 24) - 0x51) as u8;
        let layer = ((addr >> 8) & 0x7f) as u8;
        let reg = (addr & 0xff) as u8;

        let stale = current
            .as_ref()
            .is_some_and(|b| b.quadrant != quadrant || b.layer != layer);
        if stale {
            f(current.as_ref().unwrap());
            current = None;
        }
        let block = current.get_or_insert(Block {
            quadrant,
            layer,
            writes: [(0, 0); 20],
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
            0x00 => l.next = Nxtlyr::from_bits(v),
            0x04 => l.rows = Rcnt::from_bits(v),
            0x08 => l.cols = Ccnt::from_bits(v),
            0x0c => l.oned = Oned::from_bits(v),
            0x10 => l.pool_rows = Prcnt::from_bits(v),
            0x14 => l.pool_cols = Pccnt::from_bits(v),
            0x18 => l.stride = Stride::from_bits(v),
            0x1c => l.wptr[q] = WptrBase::from_bits(v),
            0x20 => l.wptr_ts = WptrToffs::from_bits(v),
            0x24 => l.wptr_moffs = Some(WptrMoffs::from_bits(v)),
            0x28 => l.wptr_choffs = WptrChoffs::from_bits(v),
            0x2c => l.rptr = RptrBase::from_bits(v),
            0x30 => l.lctl[q] = Lctl::from_bits(v),
            0x34 => l.lctl2 = Lctl2::from_bits(v),
            0x38 => l.mcnt1 = Some(Mcnt1::from_bits(v)),
            0x3c => l.mcnt2 = Mcnt2::from_bits(v),
            0x40 => l.ochan = Ochan::from_bits(v),
            0x44 => l.tptr = Tptr::from_bits(v),
            0x48 => l.ena[q] = Ena::from_bits(v),
            0x4c => l.post[q] = Post::from_bits(v),
            other => panic!("unknown layer register {other:#04x}"),
        }
    }
    l
}

#[derive(Default)]
struct Recorder {
    writes: [(u8, u32); 20],
    len: usize,
}

impl LayerSink for Recorder {
    fn write(&mut self, reg: LayerReg, bits: u32) {
        self.writes[self.len] = (reg as u8, bits);
        self.len += 1;
    }
}

/// The `LayerReg` at a byte offset
fn layer_reg(offset: u8) -> LayerReg {
    *ALL_LAYER_REGS
        .iter()
        .find(|r| **r as u8 == offset)
        .expect("offset is a layer register")
}

/// Registers the generator writes once per layer rather than per quadrant
fn is_shared(reg: u8) -> bool {
    !matches!(reg, 0x1c | 0x30 | 0x48 | 0x4c)
}

/// Every block in both networks, re-emitted from a rebuilt layer, must come
/// back byte for byte. This is what pins the emit order and the zero-write
/// suppression: `layer_from` discards order, so anything the emit path gets
/// wrong shows up here.
#[test]
fn every_block_round_trips() {
    let mut blocks = 0;
    let mut writes = 0;
    for (name, src) in [("kws20_demo", KWS20), ("cifar100", CIFAR100)] {
        for_each_block(src, |block| {
            let mut recorder = Recorder::default();
            emit_layer(&mut recorder, &layer_from(block), block.quadrant);
            assert_eq!(
                &recorder.writes[..recorder.len],
                block.writes(),
                "{name} layer {} quadrant {}",
                block.layer,
                block.quadrant
            );
            blocks += 1;
            writes += block.len;
        });
    }
    assert_eq!(blocks, 36 + 292);
    assert_eq!(writes, 497 + 4062);
}

/// The shared/per-quadrant split in `Layer` is a claim about the hardware, not
/// about this code: sixteen registers hold one value for the whole layer and
/// four vary. Checked directly against the generated poke lists.
#[test]
fn shared_registers_agree_across_quadrants() {
    for (name, src) in [("kws20_demo", KWS20), ("cifar100", CIFAR100)] {
        let mut master: Option<Block> = None;
        for_each_block(src, |block| {
            if block.quadrant == 0 {
                master = Some(*block);
                return;
            }
            let first = master.as_ref().expect("quadrant 0 block comes first");
            assert_eq!(first.layer, block.layer);

            for &(reg, value) in block.writes().iter().filter(|(r, _)| is_shared(*r)) {
                let same = first.writes().iter().find(|(r, _)| *r == reg);
                assert_eq!(
                    same.map(|(_, v)| *v),
                    Some(value),
                    "{name} layer {} register {reg:#04x} differs in quadrant {}",
                    block.layer,
                    block.quadrant
                );
            }

            // Suppression must agree too, or the register is not really shared.
            let shared_count = |b: &Block| b.writes().iter().filter(|(r, _)| is_shared(*r)).count();
            assert_eq!(
                shared_count(first),
                shared_count(block),
                "{name} layer {} emits a different set of shared registers in quadrant {}",
                block.layer,
                block.quadrant
            );
        });
    }
}

/// Both networks program all four quadrants on every layer, which is why
/// `master_only` has no golden coverage.
#[test]
fn every_layer_programs_every_quadrant() {
    for (name, src, layers) in [("kws20_demo", KWS20, 9), ("cifar100", CIFAR100, 73)] {
        let mut seen = [[false; QUADRANTS as usize]; 128];
        for_each_block(src, |block| {
            let slot = &mut seen[block.layer as usize][block.quadrant as usize];
            assert!(
                !*slot,
                "{name} layer {} quadrant {} twice",
                block.layer, block.quadrant
            );
            *slot = true;
        });
        for (index, quadrants) in seen.iter().take(layers).enumerate() {
            assert!(
                quadrants.iter().all(|q| *q),
                "{name} layer {index} is partial"
            );
        }
        assert!(seen.iter().skip(layers).all(|q| q.iter().all(|s| !*s)));
    }
}

/// The corpus the reserved-bit masks are answerable to. Every value the
/// generator emits must sit entirely inside a declared field, or the field
/// model has a gap and `validate` would reject a network that works.
///
/// Checked against all eight MSDK examples while the masks were derived; the
/// two committed here are what keeps them honest.
#[test]
fn no_shipped_value_touches_a_reserved_bit() {
    for (name, src) in [("kws20_demo", KWS20), ("cifar100", CIFAR100)] {
        for_each_block(src, |block| {
            for &(reg, value) in block.writes() {
                let reg = layer_reg(reg);
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

/// The same corpus driven through the real entry point, so the emit path and
/// the checks agree about which registers a layer has.
#[test]
fn every_layer_validates() {
    for (name, src) in [("kws20_demo", KWS20), ("cifar100", CIFAR100)] {
        for_each_block(src, |block| {
            let layer = layer_from(block);
            let layers = [layer];
            let net: Network<Direct> = Network::new(&layers, 0, 0, &[], None, &[], &[]);
            assert_eq!(
                net.validate(),
                Ok(()),
                "{name} layer {} quadrant {}",
                block.layer,
                block.quadrant
            );
        });
    }
}

/// No emitted value is zero, so the suppression in `emit` is load bearing on
/// every one of these layers rather than a rule with no instances.
#[test]
fn the_generator_never_writes_a_zero() {
    for (name, src) in [("kws20_demo", KWS20), ("cifar100", CIFAR100)] {
        for_each_block(src, |block| {
            for &(reg, value) in block.writes() {
                assert_ne!(value, 0, "{name} layer {} register {reg:#04x}", block.layer);
            }
            assert!(
                block.len < 20,
                "{name} layer {} writes every register",
                block.layer
            );
        });
    }
}
