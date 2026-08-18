//! Everything under `cnn` is tested here rather than beside the code, so the
//! modules stay readable and the fixtures are shared.

use super::boost::*;
use super::config::*;
use super::fields::*;
use super::memory::*;
use super::network::*;
use super::power::*;
use super::regs::*;
use super::run::*;
use super::validate::*;
use crate::gcr::clocks::InternalPll;
use crate::pac;
use embedded_hal::digital::{ErrorType, OutputPin};

// ---------------------------------------------------------------- addresses
// Addresses cross-checked against cnn.c from MaximSDK examples,
// but datasheet gives nothing so...
// gee.

#[test]
fn quadrant_bases() {
    assert_eq!(Quadrant::new(0).base(), 0x5100_0000);
    assert_eq!(Quadrant::new(1).base(), 0x5200_0000);
    assert_eq!(Quadrant::new(2).base(), 0x5300_0000);
    assert_eq!(Quadrant::new(3).base(), 0x5400_0000);
}

#[test]
fn bases_match_pac() {
    assert_eq!(Quadrant::new(0).base(), pac::Cnnx16_0::PTR as u32);
    assert_eq!(Quadrant::new(1).base(), pac::Cnnx16_1::PTR as u32);
    assert_eq!(Quadrant::new(2).base(), pac::Cnnx16_2::PTR as u32);
    assert_eq!(Quadrant::new(3).base(), pac::Cnnx16_3::PTR as u32);
}

#[test]
fn layer_registers() {
    let at = |q, n, r| Quadrant::new(q).layer(n).reg(r).addr();
    // kws20_demo, layer 0 quadrant 0
    assert_eq!(at(0, 0, LayerReg::Rows), 0x5110_0004);
    assert_eq!(at(0, 0, LayerReg::Oned), 0x5110_000c);
    assert_eq!(at(0, 0, LayerReg::En), 0x5110_0048);
    assert_eq!(at(0, 0, LayerReg::Post), 0x5110_004c);
    // mobilefacenet-112, layer 3 quadrant 0
    assert_eq!(at(0, 3, LayerReg::Tptr), 0x5110_0344);
    // cifar-100-mobilenet-v2, layer 72 quadrant 3
    assert_eq!(at(3, 72, LayerReg::Rows), 0x5410_4804);
    // pascalvoc-retinanetv7_3, layer 115 quadrant 2
    assert_eq!(at(2, 115, LayerReg::Rows), 0x5310_7304);
}

#[test]
fn layer_reg_offsets_are_unique_and_ordered() {
    let mut seen = [false; 0x50];
    for r in ALL_LAYER_REGS {
        let off = r as usize;
        assert!(off < 0x50 && off % 4 == 0, "bad offset {off:#x}");
        assert!(!seen[off], "duplicate offset {off:#x}");
        seen[off] = true;
    }
}

// ---------------------------------------------------------------- register fields
/// The `field!` and `flag!` macros are six lines of shift and mask shared
/// by all sixty-seven fields, so they are worth testing once rather than
/// once per register. `Rcnt` covers a field at offset 0, one abutting its
/// neighbour, a flag, and a field running to bit 31.
#[test]
fn getters_shift_and_mask() {
    let all = Rcnt::from_bits(u32::MAX);
    assert_eq!(all.cnt(), 0x7ff);
    assert_eq!(all.pad_cnt(), 0b11);
    assert!(all.pad_ena());
    assert_eq!(all.diff(), 0xffff);

    // Each field sees only its own bits.
    assert_eq!(Rcnt::from_bits(0xffff_0000).cnt(), 0);
    assert_eq!(Rcnt::from_bits(0x0000_ffff).diff(), 0);
    assert_eq!(Rcnt::from_bits(!(1 << 15)).pad_ena(), false);

    // A full-width field is not truncated, and a zero word reads as zero.
    assert_eq!(Ochan::from_bits(u32::MAX).ochan(), u32::MAX);
    assert_eq!(Rcnt::from_bits(0).diff(), 0);
}

/// mobilefacenet-112 layer 0: 112x112 input, pad 1, no pooling.
#[test]
fn mobilefacenet_layer0() {
    let rows = Rcnt::from_bits(0x0001_806f);
    assert_eq!(rows.cnt(), 111);
    assert!(rows.pad_ena());
    assert_eq!(rows.pad_cnt() + 1, 1);
    assert_eq!(rows.diff(), 1);

    let stride = Stride::from_bits(0x0000_0010);
    assert_eq!(stride.stride() + 1, 1);
    assert_eq!(stride.mp_stride(), 1);

    let wptr = WptrBase::from_bits(0x0000_2000);
    assert_eq!(wptr.offset(), 0);
    assert_eq!(wptr.instance(), 1);
    assert_eq!(wptr.group(), 0);
}

/// mobilefacenet-112 layer 1: 2x2 pool, stride 2.
#[test]
fn mobilefacenet_layer1() {
    let rows = Rcnt::from_bits(0x0072_806e);
    assert_eq!(rows.cnt(), 110);
    assert!(rows.pad_ena());
    assert_eq!(rows.diff(), 114);

    let cols = Ccnt::from_bits(0x0002_806e);
    assert_eq!(cols.cnt(), 110);
    assert_eq!(cols.diff(), 2);

    let pool = Prcnt::from_bits(0x0000_0001);
    assert_eq!(pool.pool_cnt(), 1);
    assert_eq!(pool.pool_inc(), 0);

    let stride = Stride::from_bits(0x0000_0021);
    assert_eq!(stride.stride() + 1, 2);
    assert_eq!(stride.mp_stride(), 2);
}

/// kws20_demo layer 0: Conv1d, so the column count is degenerate.
#[test]
fn kws20_layer0() {
    let rows = Rcnt::from_bits(0x0002_007f);
    assert_eq!(rows.cnt(), 127);
    assert!(!rows.pad_ena());
    assert_eq!(rows.diff(), 2);

    let cols = Ccnt::from_bits(0x0001_0000);
    assert_eq!(cols.cnt(), 0);
    assert_eq!(cols.diff(), 1);
}

/// The passthrough layer izer inserts before an average pool that follows
/// an element-wise op writes a fixed set of words.
#[test]
fn inserted_passthrough_layer() {
    let rows = Rcnt::from_bits(0x0001_0000);
    assert_eq!(rows.cnt(), 0);
    assert_eq!(rows.diff(), 1);

    let stride = Stride::from_bits(0x0000_0010);
    assert_eq!(stride.stride(), 0);
    assert_eq!(stride.mp_stride(), 1);
}

#[test]
fn nxtlyr_encodes_link_and_stop() {
    let sequential = Nxtlyr::new();
    assert!(!sequential.link_en());
    assert!(!sequential.stop());
    assert_eq!(sequential.bits(), 0);

    // Link-layer mode on a non-final layer: LINK_EN | (ll + 1).
    let linked = Nxtlyr::from_bits(0x87);
    assert_eq!(linked.bits(), 0x87);
    assert_eq!(linked.next(), 7);

    // Link-layer mode on the final layer.
    let stopped = Nxtlyr::from_bits(0x100);
    assert_eq!(stopped.bits(), 0x100);

    // The snoop-loop override force-writes a link back to layer 0.
    let loop_back = Nxtlyr::from_bits(0x80);
    assert!(loop_back.link_en());
    assert_eq!(loop_back.next(), 0);
}

/// kws20_demo layer 0: Conv1d, master quadrant, in a network with no bias.
#[test]
fn kws20_layer0_control() {
    let master = Lctl::from_bits(0x0000_eb20);
    assert!(master.unnamed5());
    assert!(master.relu());
    assert!(master.global_wptr());
    assert_eq!(master.siena(), 0b1110);
    assert!(!master.rd_ahead());

    // The same layer on a non-master quadrant differs only in SIENA.
    let slave = Lctl::from_bits(0x0000_0b20);
    assert_eq!(slave.siena(), 0);
    assert_eq!(slave.bits(), master.bits() & !(0xf << 12));

    let l2 = Lctl2::from_bits(0x0019_8011);
    assert_eq!(l2.maxpass(), 1);
    assert_eq!(l2.wptr_inc(), 1);
    assert_eq!(l2.xpch_max(), 408);

    let oned = Oned::from_bits(0x0000_1100);
    assert!(oned.oned_ena());
    assert_eq!(oned.oned_width(), 1);
    assert!(!oned.elt_ena());
    assert_eq!(oned.operands() + 1, 1);

    assert_eq!(Ochan::from_bits(0x0000_00cf).ochan() + 1, 208);
    assert_eq!(Mcnt1::from_bits(0x0000_0678).mexp_max(), 0x678);

    let ena = Ena::from_bits(0xffff_ffff);
    assert_eq!(ena.proc_ena(), 0xffff);
    assert_eq!(ena.mask_ena(), ena.proc_ena());
}

/// cifar-100-effnet2 layer 15 is the only shipped layer that sets LCTL bit
/// 29. It is a genuine depthwise broadcast, not a `shift_cnt` spill:
/// `rd_ahead` is clear, so `shift_cnt` is not written at all.
#[test]
fn lctl_bit29_is_depthwise_broadcast_in_practice() {
    let l = Lctl::from_bits(0x2088_0b20);
    assert!(!l.rd_ahead());
    assert!(l.dw_bcast());
    assert_eq!(l.cprime_max() + 1, 3);
    assert_eq!(l.rprime_max() + 1, 3);
}

/// The unguarded case `validate()` must reject: read-ahead without tcalc
/// and a large input expansion sets bit 29 through `shift_cnt` alone.
#[test]
fn lctl_shift_cnt_can_forge_broadcast() {
    let l = Lctl::from_bits((1 << 17) | (8 << 26));
    assert!(l.dw_bcast());
    assert_eq!(l.shift_cnt(), 8);
}

/// The inserted average-pool reset layer writes a fixed control word.
#[test]
fn inserted_passthrough_layer_control() {
    let l = Lctl::from_bits(0x920);
    assert!(l.unnamed5());
    assert!(l.maxpool());
    assert!(l.global_wptr());
    assert!(!l.pool_ena());
    assert_eq!(l.siena(), 0);

    let ena = Ena::from_bits(1);
    assert_eq!(ena.proc_ena(), 1);
    assert_eq!(ena.mask_ena(), 0);
}

/// Element-wise layers encode the operand count minus one.
#[test]
fn eltwise_operands() {
    let o = Oned::from_bits(0x0004_6003);
    assert!(o.elt_ena());
    assert!(!o.oned_ena());
    assert_eq!(o.elt_fn(), 1);
    assert_eq!(o.operands() + 1, 2);
    assert!(!o.pool_first());

    // Same layer with pooling ahead of the element-wise op.
    let pooled = Oned::from_bits(0x0005_6003);
    assert!(pooled.pool_first());
    assert_eq!(pooled.bits(), o.bits() | (1 << 16));
}

const POSTS: [u32; 16] = [
    0x0000_1000,
    0x0000_113e,
    0x0000_15b0,
    0x0000_2000,
    0x0000_3100,
    0x0000_5430,
    0x0000_73f0,
    0x0000_8000,
    0x0002_2000,
    0x0002_34c4,
    0x0002_5000,
    0x0002_73b0,
    0x0100_0000,
    0x0300_0000,
    0x1000_1200,
    0x4100_d004,
];

/// The field the plan singles out as most error-prone: five bits of signed
/// magnitude, not two's complement.
#[test]
fn output_shift_is_signed_magnitude() {
    // The worked example from the specification.
    let p = Post::from_bits((3 << 13) | (1 << 17));
    assert_eq!(p.scale_mag(), 3);
    assert!(p.scale_dir());
    assert_eq!(p.bits() >> 13, 0b1_0011);
    assert_ne!(p.bits() >> 13, 0b1_1101, "encoded as two's complement");
    assert_eq!(p.output_shift(), -3);

    for shift in -15..=15i32 {
        let raw = (shift.unsigned_abs() << 13) | ((shift < 0) as u32) << 17;
        assert_eq!(Post::from_bits(raw).output_shift(), shift, "shift {shift}");
    }

    // Every scale actually shipped, decoded from the golden words.
    assert_eq!(Post::from_bits(0x0000_8000).output_shift(), 4);
    assert_eq!(Post::from_bits(0x0000_73f0).output_shift(), 3);
    assert_eq!(Post::from_bits(0x0002_2000).output_shift(), -1);
    assert_eq!(Post::from_bits(0x0002_5000).output_shift(), -2);
    assert_eq!(Post::from_bits(0x0002_73b0).output_shift(), -3);
    assert_eq!(Post::from_bits(0x4100_d004).output_shift(), 6);
}

#[test]
fn shift_direction_matches_the_raw_bit() {
    assert_eq!(Post::from_bits(2 << 13).shift_dir(), ShiftDir::Left);
    assert_eq!(
        Post::from_bits((2 << 13) | (1 << 17)).shift_dir(),
        ShiftDir::Right
    );
    assert!(Post::from_bits(1 << 17).scale_dir());
}

/// cifar-100-effnet2 layer 15: depthwise, so `dw_ena` and `ts_ena` are
/// written together.
#[test]
fn effnet2_layer15_post() {
    let p = Post::from_bits(0x4100_3000);
    assert!(p.dw_ena());
    assert!(p.ts_ena(), "dw_ena is always written with ts_ena");
    assert!(p.bias_en());
    assert_eq!(p.bias_addr(), 0);
    assert_eq!(p.output_shift(), 1);
    assert_eq!(p.weight_scale(), WeightScale::Bits8);
}

/// The inserted average-pool reset layer writes `POST = 0x0300_0000`.
#[test]
fn inserted_passthrough_layer_post() {
    let p = Post::from_bits(0x0300_0000);
    assert!(p.ts_ena());
    assert!(p.onexone_ena());
    assert!(!p.bias_en());
    assert_eq!(p.output_shift(), 0, "passthrough must not shift");
}

#[test]
fn weight_scale_codes_match_the_generator() {
    assert_eq!(WeightScale::from_code(0), WeightScale::Bits8);
    assert_eq!(WeightScale::from_code(1), WeightScale::Bits1);
    assert_eq!(WeightScale::from_code(2), WeightScale::Bits2);
    assert_eq!(WeightScale::from_code(3), WeightScale::Bits4);
    for s in [
        WeightScale::Bits8,
        WeightScale::Bits1,
        WeightScale::Bits2,
        WeightScale::Bits4,
    ] {
        assert_eq!(Post::from_bits((s as u32) << 22).weight_scale(), s);
    }
    // Every shipped network uses 8-bit weights.
    for bits in POSTS {
        assert_eq!(Post::from_bits(bits).weight_scale(), WeightScale::Bits8);
    }
}

#[test]
fn eltwise_codes_match_the_generator() {
    assert_eq!(EltwiseFn::from_code(0b00), EltwiseFn::Sub);
    assert_eq!(EltwiseFn::from_code(0b01), EltwiseFn::Add);
    assert_eq!(EltwiseFn::from_code(0b10), EltwiseFn::Or);
    assert_eq!(EltwiseFn::from_code(0b11), EltwiseFn::Xor);
    // The shipped element-wise layers are adds.
    assert_eq!(Oned::from_bits(0x0004_6003).eltwise_fn(), EltwiseFn::Add);
    for f in [
        EltwiseFn::Sub,
        EltwiseFn::Add,
        EltwiseFn::Or,
        EltwiseFn::Xor,
    ] {
        assert_eq!(Oned::from_bits((f as u32) << 14).eltwise_fn(), f);
    }
}

#[test]
fn pool_mode_follows_maxpool() {
    assert_eq!(Lctl::from_bits(0x920).pool_mode(), PoolMode::Max);
    assert_eq!(Lctl::new().pool_mode(), PoolMode::Avg);
    assert_eq!(Lctl::from_bits(1 << 8).pool_mode(), PoolMode::Max);
}

/// Activation spans two registers, so it round-trips through both.
#[test]
fn activation_spans_lctl_and_post() {
    let lctl = Lctl::from_bits(0x0000_eb20);
    let post = Post::from_bits(0x0000_2000);
    assert_eq!(Activation::decode(lctl, post), Some(Activation::Relu));

    for (l, p, act) in [
        (0, 0, Activation::None),
        (1 << 9, 0, Activation::Relu),
        (0, 1 << 26, Activation::Abs),
    ] {
        let decoded = Activation::decode(Lctl::from_bits(l), Post::from_bits(p));
        assert_eq!(decoded, Some(act));
    }

    // Both bits set is not a valid encoding.
    let both = Lctl::from_bits(1 << 9);
    let abs = Post::from_bits(1 << 26);
    assert_eq!(Activation::decode(both, abs), None);
}

fn assert_reg<R: LayerRegister>(expected: LayerReg, bits: u32) {
    assert_eq!(R::REG, expected);
    assert_eq!(<R as LayerRegister>::from_bits(bits).bits(), bits);
}

/// Pins the value-type to register-slot mapping one type at a time. The
/// table in `layer_registers!` is the only place it is written down, so a
/// swapped pair there would otherwise go unnoticed.
#[test]
fn every_type_maps_to_its_own_slot() {
    assert_reg::<Nxtlyr>(LayerReg::Next, 0x87);
    assert_reg::<Rcnt>(LayerReg::Rows, 0x0002_007f);
    assert_reg::<Ccnt>(LayerReg::Cols, 0x0001_0000);
    assert_reg::<Oned>(LayerReg::Oned, 0x0000_1100);
    assert_reg::<Prcnt>(LayerReg::PoolRows, 0x1);
    assert_reg::<Pccnt>(LayerReg::PoolCols, 0x1);
    assert_reg::<Stride>(LayerReg::Stride, 0x20);
    assert_reg::<WptrBase>(LayerReg::Wptr, 0x800);
    assert_reg::<WptrToffs>(LayerReg::WptrTs, 0x1);
    assert_reg::<WptrMoffs>(LayerReg::WptrMask, 0x8000);
    assert_reg::<WptrChoffs>(LayerReg::WptrMp, 0x1);
    assert_reg::<RptrBase>(LayerReg::Rptr, 0x2000);
    assert_reg::<Lctl>(LayerReg::Lctl, 0x0000_eb20);
    assert_reg::<Lctl2>(LayerReg::Lctl2, 0x0019_8011);
    assert_reg::<Mcnt1>(LayerReg::Mcnt, 0x678);
    assert_reg::<Mcnt2>(LayerReg::Moffs, 0x1200);
    assert_reg::<Ochan>(LayerReg::Ochan, 0xcf);
    assert_reg::<Tptr>(LayerReg::Tptr, 0x0140_027f);
    assert_reg::<Ena>(LayerReg::En, 0xffff_ffff);
    assert_reg::<Post>(LayerReg::Post, 0x0000_3000);
}

#[test]
fn typed_registers_cover_every_slot_once() {
    assert_eq!(ALL_TYPED_REGS.len(), crate::cnn::regs::ALL_LAYER_REGS.len());
    for expected in crate::cnn::regs::ALL_LAYER_REGS {
        let count = ALL_TYPED_REGS.iter().filter(|r| **r == expected).count();
        assert_eq!(count, 1, "{expected:?} has {count} value types");
    }
}

/// Every `DECLARED_BITS` constant is a hand-written literal, so this pins
/// each one to the fields actually declared above it. A bit is declared
/// exactly when toggling it changes what some getter reports, which the
/// `Debug` impl renders. Catches a mask that drifts after a field is
/// added, moved or widened, in either direction.
///
/// Toggled against both an all-zero and an all-ones word because some
/// `Debug` impls are conditional: `Nxtlyr` renders `next` only when
/// `link_en` is set, so probing up from zero alone would miss it.
#[test]
fn declared_bits_match_the_getters() {
    for reg in ALL_TYPED_REGS {
        let (zero, ones) = (rendered(reg, 0), rendered(reg, u32::MAX));
        for bit in 0..32 {
            let up = rendered(reg, 1 << bit);
            let down = rendered(reg, u32::MAX ^ (1 << bit));
            let visible = up.as_str() != zero.as_str() || down.as_str() != ones.as_str();
            let declared = reserved_bits(reg, 1 << bit) == 0;
            assert_eq!(
                visible,
                declared,
                "{reg:?} bit {bit} is {} but reads back {}",
                if declared { "declared" } else { "reserved" },
                if visible { "visible" } else { "as zero" }
            );
        }
    }
}

/// A register whose fields tile the whole word can have nothing reserved.
#[test]
fn full_width_registers_reserve_nothing() {
    for reg in [
        LayerReg::Stride,
        LayerReg::WptrTs,
        LayerReg::WptrMask,
        LayerReg::WptrMp,
        LayerReg::Rptr,
        LayerReg::Ochan,
        LayerReg::Tptr,
        LayerReg::En,
        LayerReg::Post,
    ] {
        assert_eq!(reserved_bits(reg, u32::MAX), 0, "{reg:?}");
    }
}

/// The registers where the check can actually fire, with the reserved
/// windows spelled out.
#[test]
fn reserved_windows_are_where_expected() {
    for (reg, bits) in [
        (LayerReg::Next, 0xffff_fe00u32),
        (LayerReg::Rows, 0x0000_1800),
        (LayerReg::Cols, 0x0000_1800),
        (LayerReg::Oned, 0xffc0_0000),
        (LayerReg::PoolRows, 0xffff_ff00),
        (LayerReg::PoolCols, 0xffff_ff00),
        (LayerReg::Wptr, 0xffe0_0000),
        (LayerReg::Lctl, 0x8000_041f),
        (LayerReg::Lctl2, 0xffe0_0000),
        (LayerReg::Mcnt, 0xfff8_0000),
        (LayerReg::Moffs, 0xfff8_0000),
    ] {
        assert_eq!(reserved_bits(reg, u32::MAX), bits, "{reg:?}");
        assert_eq!(reserved_bits(reg, !bits), 0, "{reg:?}");
    }
}

/// A `core::fmt` sink, so the decoder can be checked without `std`.
struct Buf {
    data: [u8; 512],
    len: usize,
}

impl Buf {
    const fn new() -> Self {
        Self {
            data: [0; 512],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.data[..self.len]).unwrap()
    }
}

impl core::fmt::Write for Buf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let n = bytes.len().min(self.data.len() - self.len);
        self.data[self.len..self.len + n].copy_from_slice(&bytes[..n]);
        self.len += n;
        Ok(())
    }
}

fn rendered(reg: LayerReg, bits: u32) -> Buf {
    use core::fmt::Write;
    let mut buf = Buf::new();
    with_decoded(reg, bits, |v| write!(buf, "{v:?}").unwrap());
    buf
}

#[test]
fn decoder_visits_every_register() {
    let mut visited = 0;
    for reg in crate::cnn::regs::ALL_LAYER_REGS {
        with_decoded(reg, 0, |_| visited += 1);
    }
    assert_eq!(visited, 20);
}

/// The decoder must produce the register's own type, not a same-shaped
/// neighbour. `Rcnt` and `Ccnt` have identical layouts, so only the name
/// distinguishes them.
#[test]
fn decoder_names_the_right_type() {
    assert!(rendered(LayerReg::Rows, 0x0002_007f)
        .as_str()
        .starts_with("Rcnt"));
    assert!(rendered(LayerReg::Cols, 0x0001_0000)
        .as_str()
        .starts_with("Ccnt"));
    assert!(rendered(LayerReg::PoolRows, 1)
        .as_str()
        .starts_with("Prcnt"));
    assert!(rendered(LayerReg::PoolCols, 1)
        .as_str()
        .starts_with("Pccnt"));
    assert!(rendered(LayerReg::Mcnt, 0x678)
        .as_str()
        .starts_with("Mcnt1"));
    assert!(rendered(LayerReg::Moffs, 0x1200)
        .as_str()
        .starts_with("Mcnt2"));
    assert!(rendered(LayerReg::En, 0xffff_ffff)
        .as_str()
        .starts_with("Ena"));
}

/// The payoff the plan is after: a dumped word reads as fields, not hex.
#[test]
fn decoder_output_is_decoded_not_hex() {
    let post = rendered(LayerReg::Post, 0x0002_73b0);
    let text = post.as_str();
    assert!(text.contains("output_shift: -3"), "{text}");
    assert!(text.contains("bias_en: true"), "{text}");

    let lctl = rendered(LayerReg::Lctl, 0x0000_eb20);
    assert!(lctl.as_str().contains("relu: true"), "{}", lctl.as_str());
}

// ---------------------------------------------------------------- memories
// Addresses cross-checked against generated cnn.c and weights.h in the
// MaximSDK MAX78002 CNN examples, and against the C_* base constants in
// ai8x-synthesis izer/tornadocnn.py.

#[test]
fn kernel_addresses() {
    // weights.h KERNELS blobs run from 0x51400000 to 0x545e0000.
    assert_eq!(kernel_addr(0, 0), 0x5140_0000);
    assert_eq!(kernel_addr(0, 1), 0x5142_0000);
    assert_eq!(kernel_addr(0, 15), 0x515e_0000);
    assert_eq!(kernel_addr(3, 15), 0x545e_0000);
}

#[test]
fn bias_addresses() {
    // cnn_load_bias writes 0x51180000 through 0x54180000.
    assert_eq!(bias_addr(0), 0x5118_0000);
    assert_eq!(bias_addr(3), 0x5418_0000);
}

#[test]
fn tram_addresses() {
    // C_TRAM_BASE = C_CNN_BASE + 0x200000, processor stride 0x10000.
    assert_eq!(tram_addr(0, 0), 0x5120_0000);
    assert_eq!(tram_addr(0, 15), 0x512f_0000);
    assert_eq!(tram_addr(3, 0), 0x5420_0000);
    // 16 processors of 0x10000 fit between TRAM and the kernel memory.
    assert!(tram_addr(0, 15) + 0x1_0000 <= kernel_addr(0, 0));
}

#[test]
fn data_addresses() {
    // kws20_demo cnn_load_data walks 0x51800000..0x51860000 per quadrant.
    assert_eq!(data_addr(0, 0), 0x5180_0000);
    assert_eq!(data_addr(0, 3), 0x5186_0000);
    // imagenet loads its input at 0x54860000.
    assert_eq!(data_addr(3, 3), 0x5486_0000);
}

/// The documented per-processor capacities must sum to the total that
/// `ai8xize.py` reports.
#[test]
fn kernel_capacity_totals_match_datasheet() {
    let kernels: u32 = (0..QUADRANTS)
        .map(|_| {
            (0..PROCESSORS_PER_QUADRANT)
                .map(kernel_capacity)
                .sum::<u32>()
        })
        .sum();
    assert_eq!(kernels * 9, TOTAL_KERNEL_BYTES);
}

#[test]
fn data_window_holds_four_processor_slots() {
    assert_eq!(
        DATA_INSTANCE_STRIDE_WORDS * (DATA_INSTANCES_PER_QUADRANT as u32) * 4,
        DATA_WINDOW_BYTES
    );
    // Backed memory is smaller than the address space it sits in.
    assert!(DATA_INSTANCE_WORDS < DATA_INSTANCE_STRIDE_WORDS);
    // 64 processors x 5120 words x 4 bytes is the advertised 1.31MB.
    let total =
        DATA_INSTANCE_WORDS as u64 * (QUADRANTS as u64) * (PROCESSORS_PER_QUADRANT as u64) * 4;
    assert_eq!(total, 1_310_720);
}

/// `imagenet` writes 12544 words to one instance in a single call. Any
/// bound tighter than the window would reject it.
#[test]
fn bound_admits_the_largest_shipped_input_load() {
    assert!(12544 * 4 <= DATA_WINDOW_BYTES);
    assert!(12544 > DATA_INSTANCE_WORDS);
}

/// Record addresses taken from the `KERNELS` blobs in `weights.h`. Each
/// record is `(address, length, data...)`, and the address decomposes into
/// a quadrant, a processor and a byte offset.
#[test]
fn kernel_burst_addresses_match_the_weight_blobs() {
    // kws20_demo, the first five records of processor 0.
    assert_eq!(kernel_addr(0, 0) + 0x000, 0x5140_0000);
    assert_eq!(kernel_addr(0, 0) + 0x210, 0x5140_0210);
    assert_eq!(kernel_addr(0, 0) + 0x2b0, 0x5140_02b0);
    assert_eq!(kernel_addr(0, 0) + 0x450, 0x5140_0450);
    assert_eq!(kernel_addr(0, 0) + 0x500, 0x5140_0500);
    // The next processor starts a fresh window.
    assert_eq!(kernel_addr(0, 1) + 0x000, 0x5142_0000);
    assert_eq!(kernel_addr(0, 1) + 0x210, 0x5142_0210);
}

/// The window is far larger than any shipped network needs, so the bound
/// is generous by design. `pascalvoc-retinanetv7_3` has the longest single
/// burst at 5686 words; `imagenet-riscv` reaches the highest end offset.
#[test]
fn bound_admits_the_largest_shipped_weight_bursts() {
    assert!(kernel_burst_fits(0, 5686));
    assert!(kernel_burst_fits(0x5b7c - 4611 * 4, 4611));
    // Roughly a fifth of the window is in use at worst.
    assert!(0x5b7c * 5 < KERNEL_WINDOW_BYTES);
}

/// Bias sizes differ per quadrant. These are every size the shipped
/// networks use; `mobilefacenet-112` has the largest at 1840.
#[test]
fn bound_admits_every_shipped_bias_table() {
    for entries in [
        256, 229, 224, // kinetics
        856, 832, 894, // pascalvoc-retinanetv7_3
        1392, 1296, 1268, 1280, // cifar-100-effnet2
        1384, 1376, // imagenet
        1632, 1628, 1620, // cifar-100-mobilenet-v2-0.75
        1840, 1776, // mobilefacenet-112
    ] {
        assert!(bias_fits(entries), "{entries} entries rejected");
    }
    assert!(bias_fits(BIAS_ENTRIES as usize));
    assert!(!bias_fits(BIAS_ENTRIES as usize + 1));
}

/// Each entry takes a whole word, so the region is four times the entry
/// count in bytes and `POST.bias_addr` can reach all of it.
#[test]
fn bias_region_is_one_word_per_entry() {
    assert_eq!(bias_addr(0) + BIAS_ENTRIES * 4, bias_addr(0) + 8192);
    // POST.bias_addr is 12 bits, so it addresses more words than exist.
    assert!(BIAS_ENTRIES <= 1 << 12);
    // The region fits between the bias base and the TRAM.
    assert!(bias_addr(0) + BIAS_ENTRIES * 4 <= tram_addr(0, 0));
}

/// `kws20_demo` unloads 21 words of 32-bit output. The generated sequence
/// reads four words from each of six places; these are its addresses.
#[test]
fn output_addresses_match_the_generated_unload() {
    let at = |q, i, word: u32| data_addr(q, i) + word * 4;
    assert_eq!(at(0, 0, 2048), 0x5180_2000);
    assert_eq!(at(0, 1, 2048), 0x5182_2000);
    assert_eq!(at(0, 2, 2048), 0x5184_2000);
    assert_eq!(at(0, 3, 2048), 0x5186_2000);
    assert_eq!(at(1, 0, 2048), 0x5280_2000);
    assert_eq!(at(1, 1, 2048), 0x5282_2000);
}

/// `mobilefacenet-112` unloads 8-bit output. Its C loop does
/// `addr += 0x8000` on a `uint32_t *`, which advances 0x20000 **bytes** —
/// one whole instance — so the walk crosses instances rather than words.
#[test]
fn eight_bit_output_walks_instances_not_words() {
    let at = |q, i, word: u32| data_addr(q, i) + word * 4;
    assert_eq!(at(0, 0, 10240), 0x5180_a000);
    // One `addr += 0x8000` step in the generated loop.
    assert_eq!(at(0, 1, 10240), at(0, 0, 10240) + 0x2_0000);
    assert_eq!(at(0, 2, 10240), 0x5184_a000);
    assert_eq!(at(1, 0, 10240), 0x5280_a000);

    // The offset sits past the first processor's backed words but inside
    // the window, which is why the bound is the window.
    assert!(10240 > DATA_INSTANCE_WORDS);
    assert!(10240 * 4 < DATA_WINDOW_BYTES);
}

/// The generated unload emits `val & 0xff`, then `>> 8`, `>> 16`, `>> 24`.
#[test]
fn word_unpacks_least_significant_byte_first() {
    let val: u32 = 0x4433_2211;
    assert_eq!(val.to_le_bytes(), [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(
        val.to_le_bytes(),
        [
            (val & 0xff) as u8,
            ((val >> 8) & 0xff) as u8,
            ((val >> 16) & 0xff) as u8,
            ((val >> 24) & 0xff) as u8,
        ]
    );
}

#[test]
fn bound_rejects_bursts_that_leave_the_window() {
    assert!(kernel_burst_fits(0, (KERNEL_WINDOW_BYTES / 4) as usize));
    assert!(!kernel_burst_fits(
        0,
        (KERNEL_WINDOW_BYTES / 4) as usize + 1
    ));
    assert!(!kernel_burst_fits(KERNEL_WINDOW_BYTES, 1));
    assert!(kernel_burst_fits(KERNEL_WINDOW_BYTES - 4, 1));
    // The arithmetic must not wrap.
    assert!(!kernel_burst_fits(u32::MAX, 1));
}

// ---------------------------------------------------------------- descriptors
fn synthetic_layer() -> Layer {
    Layer {
        next: Nxtlyr::new(),
        rows: Rcnt::from_bits(0x0001_0000),
        cols: Ccnt::from_bits(0x0001_0000),
        oned: Oned::from_bits(0x0000_0100),
        pool_rows: Prcnt::new(),
        pool_cols: Pccnt::new(),
        stride: Stride::from_bits(0x0000_0010),
        wptr_ts: WptrToffs::new(),
        wptr_moffs: None,
        wptr_choffs: WptrChoffs::new(),
        rptr: RptrBase::from_bits(0x2000),
        lctl2: Lctl2::new(),
        mcnt1: None,
        mcnt2: Mcnt2::new(),
        ochan: Ochan::new(),
        tptr: Tptr::new(),
        lctl: [
            Lctl::from_bits(0x920),
            Lctl::new(),
            Lctl::new(),
            Lctl::new(),
        ],
        post: [
            Post::from_bits(0x0300_0000),
            Post::new(),
            Post::new(),
            Post::new(),
        ],
        wptr: [WptrBase::from_bits(0x2000); 4],
        ena: [Ena::from_bits(1), Ena::new(), Ena::new(), Ena::new()],
        master_only: true,
    }
}

#[test]
fn recognises_the_inserted_passthrough_layer() {
    assert!(synthetic_layer().is_synthetic());

    // A real layer with any of the fixed words changed is not synthetic.
    let mut real = synthetic_layer();
    real.lctl[0] = Lctl::from_bits(0x0000_eb20);
    assert!(!real.is_synthetic());

    let mut real = synthetic_layer();
    real.rows = Rcnt::from_bits(0x0002_007f);
    assert!(!real.is_synthetic());
}

/// Passthrough layers skip these two writes entirely
#[test]
fn passthrough_layers_omit_two_registers() {
    let layer = synthetic_layer();
    assert!(layer.wptr_moffs.is_none());
    assert!(layer.mcnt1.is_none());
}

/// `kws20_demo` produces 21 words of 32-bit output from six regions:
/// five of four words and one of one.
#[test]
fn output_words_sums_the_regions() {
    const OUTPUT: [OutputRegion; 6] = [
        OutputRegion {
            quadrant: 0,
            instance: 0,
            word: 2048,
            len: 4,
        },
        OutputRegion {
            quadrant: 0,
            instance: 1,
            word: 2048,
            len: 4,
        },
        OutputRegion {
            quadrant: 0,
            instance: 2,
            word: 2048,
            len: 4,
        },
        OutputRegion {
            quadrant: 0,
            instance: 3,
            word: 2048,
            len: 4,
        },
        OutputRegion {
            quadrant: 1,
            instance: 0,
            word: 2048,
            len: 4,
        },
        OutputRegion {
            quadrant: 1,
            instance: 1,
            word: 2048,
            len: 1,
        },
    ];
    let net: Network = Network::new(&[], 0, 0, &[], None, &[], &OUTPUT);
    assert_eq!(net.output_words(), 21);

    // mobilefacenet-112: 64 channels of 8-bit output as 16 single words,
    // one per instance across all four quadrants.
    const BYTES: [OutputRegion; 16] = [OutputRegion {
        quadrant: 0,
        instance: 0,
        word: 10240,
        len: 1,
    }; 16];
    let net: Network = Network::new(&[], 0, 0, &[], None, &[], &BYTES);
    assert_eq!(net.output_words(), 16);
    assert_eq!(net.output_words() * 4, 64);
}

/// `kinetics` spreads its input over twelve regions of 7200 words, four
/// instances in each of three quadrants.
#[test]
fn input_words_sums_the_regions() {
    let regions: [InputRegion; 12] = core::array::from_fn(|i| InputRegion {
        quadrant: (i / 4) as u8,
        instance: (i % 4) as u8,
        word: 960,
        len: 7200,
    });
    let net: Network = Network::new(&[], 0, 0, &[], None, &regions, &[]);
    assert_eq!(net.input_words(), 12 * 7200);

    let net: Network = Network::new(&[], 0, 0, &[], None, &[], &[]);
    assert_eq!(net.input_words(), 0);
}

/// The descriptors must resolve to the addresses the generated
/// `load_input` pokes: `cifar-100-mobilenet-v2-0.75` loads 1024 words at
/// `0x51800000`, `imagenet` 12544 at `0x54860000`, and `kinetics` twelve
/// runs of 7200 at word 960 of each instance.
#[test]
fn input_regions_match_the_generated_addresses() {
    let addr = |r: InputRegion| data_addr(r.quadrant, r.instance) + r.word as u32 * 4;

    for (quadrant, instance, word, expected) in [
        (0u8, 0u8, 0u16, 0x5180_0000u32),
        (3, 3, 0, 0x5486_0000),
        (0, 0, 960, 0x5180_0f00),
        (0, 3, 960, 0x5186_0f00),
        (1, 0, 960, 0x5280_0f00),
        (2, 3, 960, 0x5386_0f00),
    ] {
        assert_eq!(
            addr(InputRegion {
                quadrant,
                instance,
                word,
                len: 1
            }),
            expected
        );
    }
}

/// `new` is the const-context form of `Default`, and generated networks
/// use it with struct update syntax, so the two must not drift.
#[test]
fn new_matches_default() {
    assert_eq!(Layer::new(), Layer::default());
}

// ---------------------------------------------------------------- validation
fn net(layers: &[Layer]) -> Network<'_> {
    Network::new(
        layers,
        0,
        (layers.len() as u8).saturating_sub(1),
        &[],
        None,
        &[],
        &[],
    )
}

#[test]
fn a_blank_layer_is_valid() {
    assert_eq!(net(&[Layer::default()]).validate(), Ok(()));
}

/// `kws20_demo` layer 0: `LCTL = 0xeb20` on the master naming quadrants
/// 1-3, `0x0b20` elsewhere, and `ENA = 0xffffffff`.
#[test]
fn the_shipped_shape_validates() {
    let mut layer = Layer::default();
    layer.lctl = [
        Lctl::from_bits(0x0000_eb20),
        Lctl::from_bits(0x0000_0b20),
        Lctl::from_bits(0x0000_0b20),
        Lctl::from_bits(0x0000_0b20),
    ];
    layer.ena = [Ena::from_bits(0xffff_ffff); 4];
    assert_eq!(net(&[layer]).validate(), Ok(()));
}

/// Every `ENA` value in the eight shipped examples is one of these, and
/// all of them satisfy the rule.
#[test]
fn shipped_enable_words_pass() {
    for bits in [
        0x0000_000fu32,
        0x0000_00ff,
        0x0000_0fff,
        0x0000_ffff,
        0x0007_0007,
        0x000f_000f,
        0x00ff_00ff,
        0x0fff_0fff,
        0x7000_7000,
        0xf000_f000,
        0xffff_ffff,
    ] {
        let mut layer = Layer::default();
        layer.ena = [Ena::from_bits(bits); 4];
        assert_eq!(net(&[layer]).validate(), Ok(()), "{bits:#010x}");
    }
}

/// A mask enable that is neither zero nor a mirror means the weight
/// memories armed do not match the processors doing the work.
#[test]
fn mismatched_mask_enables_are_rejected() {
    let mut layer = Layer::default();
    layer.ena = [Ena::from_bits(0x000f_00ff); 4];
    assert_eq!(
        net(&[layer]).validate(),
        Err(Invalid::MaskEnables {
            layer: 0,
            quadrant: 0
        })
    );
}

/// `SIENA` lives only in the master's word, and never names the master.
#[test]
fn source_enables_belong_to_the_master() {
    let mut layer = Layer::default();
    layer.lctl[1] = Lctl::from_bits(0b1110 << 12);
    assert_eq!(
        net(&[layer]).validate(),
        Err(Invalid::SourceEnables {
            layer: 0,
            quadrant: 1
        })
    );

    let mut layer = Layer::default();
    layer.lctl[0] = Lctl::from_bits(0b1111 << 12);
    assert_eq!(
        net(&[layer]).validate(),
        Err(Invalid::SourceEnables {
            layer: 0,
            quadrant: 0
        })
    );
}

/// Spec 8.4. Only the three conditions together are a fault: 688 shipped
/// layers set `rd_ahead` with `tcalc` clear, and 180 carry a shift count
/// of 8 or more, but no shipped layer does both.
#[test]
fn read_ahead_shift_collision_is_rejected() {
    let bad = Lctl::from_bits((1 << 17) | (8 << 26));
    let mut layer = Layer::default();
    layer.lctl = [bad; 4];
    assert_eq!(
        net(&[layer]).validate(),
        Err(Invalid::ShiftCountCollision {
            layer: 0,
            quadrant: 0
        })
    );

    // Seven fits in the three bits below DW_BCAST.
    let mut layer = Layer::default();
    layer.lctl = [Lctl::from_bits((1 << 17) | (7 << 26)); 4];
    assert_eq!(net(&[layer]).validate(), Ok(()));

    // With tcalc the field is a different, smaller quantity.
    let mut layer = Layer::default();
    layer.lctl = [bad; 4];
    layer.post = [Post::from_bits(1 << 31); 4];
    assert_eq!(net(&[layer]).validate(), Ok(()));

    // Read-ahead is what makes the field mean in_expand at all.
    let mut layer = Layer::default();
    layer.lctl = [Lctl::from_bits(15 << 26); 4];
    assert_eq!(net(&[layer]).validate(), Ok(()));
}

#[test]
fn layer_bounds_are_checked() {
    let layers = [Layer::default(), Layer::default()];
    let n: Network = Network::new(&layers, 1, 0, &[], None, &[], &[]);
    assert_eq!(
        n.validate(),
        Err(Invalid::LayerRange {
            first: 1,
            last: 0,
            layers: 2
        })
    );

    let n: Network = Network::new(&layers, 0, 2, &[], None, &[], &[]);
    assert!(matches!(n.validate(), Err(Invalid::LayerRange { .. })));
}

#[test]
fn weight_regions_are_bounds_checked() {
    const DATA: [u32; 4] = [0; 4];

    let bad = [WeightRegion {
        quadrant: 4,
        processor: 0,
        offset: 0,
        data: &DATA,
    }];
    let n: Network = Network::new(&[], 0, 0, &bad, None, &[], &[]);
    assert_eq!(n.validate(), Err(Invalid::WeightTarget { region: 0 }));

    // One word short of the window is fine; the window itself is not.
    let bad = [WeightRegion {
        quadrant: 0,
        processor: 0,
        offset: crate::cnn::memory::KERNEL_WINDOW_BYTES - 4,
        data: &DATA,
    }];
    let n: Network = Network::new(&[], 0, 0, &bad, None, &[], &[]);
    assert_eq!(n.validate(), Err(Invalid::WeightOverrun { region: 0 }));
}

#[test]
fn data_regions_are_bounds_checked() {
    let bad = [InputRegion {
        quadrant: 0,
        instance: 4,
        word: 0,
        len: 1,
    }];
    let n: Network = Network::new(&[], 0, 0, &[], None, &bad, &[]);
    assert_eq!(
        n.validate(),
        Err(Invalid::DataTarget {
            region: 0,
            input: true
        })
    );

    let bad = [OutputRegion {
        quadrant: 0,
        instance: 0,
        word: 32767,
        len: 4,
    }];
    let n: Network = Network::new(&[], 0, 0, &[], None, &[], &bad);
    assert_eq!(
        n.validate(),
        Err(Invalid::DataOverrun {
            region: 0,
            input: false
        })
    );
}

#[test]
fn bias_tables_are_bounds_checked() {
    const BIG: [u8; 2049] = [0; 2049];
    const TABLES: [&[u8]; 4] = [&[], &BIG, &[], &[]];
    let n: Network = Network::new(&[], 0, 0, &[], Some(&TABLES), &[], &[]);
    assert_eq!(n.validate(), Err(Invalid::BiasOverrun { quadrant: 1 }));
}

/// `RCNT` bits 11 and 12 sit between `CNT` and `PAD_CNT`. A count that
/// overran its eleven bits would land exactly here, and the getter would
/// still report an in-range `cnt` - which is why the range check the plan
/// asked for cannot see this and the reserved check can.
#[test]
fn reserved_bits_are_rejected() {
    let mut layer = Layer::default();
    layer.rows = Rcnt::from_bits(0x0000_0800);
    assert_eq!(
        net(&[layer.clone()]).validate(),
        Err(Invalid::ReservedBits {
            layer: 0,
            quadrant: 0,
            reg: LayerReg::Rows,
            bits: 0x0000_0800,
        })
    );
    assert_eq!(Rcnt::from_bits(0x0000_0800).cnt(), 0);

    // The same word without the stray bit is fine.
    layer.rows = Rcnt::from_bits(0x0002_007f);
    assert_eq!(net(&[layer]).validate(), Ok(()));
}

/// The check runs through `emit_layer`, so it sees exactly the registers
/// that would be written and skips the ones suppression drops.
#[test]
fn suppressed_registers_are_not_checked() {
    // Reserved bits alone cannot make a register non-zero and then vanish,
    // but an absent optional register must not be checked at all.
    let mut layer = Layer::default();
    layer.mcnt1 = Some(Mcnt1::from_bits(0xfff8_0000));
    assert!(matches!(
        net(&[layer.clone()]).validate(),
        Err(Invalid::ReservedBits {
            reg: LayerReg::Mcnt,
            ..
        })
    ));

    layer.mcnt1 = None;
    assert_eq!(net(&[layer]).validate(), Ok(()));
}

/// Per-quadrant registers are checked in their own quadrant only.
#[test]
fn reserved_bits_are_reported_per_quadrant() {
    let mut layer = Layer::default();
    layer.lctl[2] = Lctl::from_bits(0x8000_0000);
    assert_eq!(
        net(&[layer]).validate(),
        Err(Invalid::ReservedBits {
            layer: 0,
            quadrant: 2,
            reg: LayerReg::Lctl,
            bits: 0x8000_0000,
        })
    );
}

// ---------------------------------------------------------------- clock and power
#[test]
fn the_divider_defaults_to_the_reset_value() {
    // PCLKDIV.CNNCLKDIV comes out of reset at div-by-2, not div-by-1.
    assert_eq!(CnnClockDiv::default(), CnnClockDiv::Div2);
}

/// The PLL's CNN branch runs at twice the system branch, so the source
/// frequency must not be taken from the `Clock` itself.
#[test]
fn pll_source_uses_the_cnn_branch() {
    assert_eq!(InternalPll::CNN_FREQUENCY, MAX_PIPELINED_FREQUENCY);
    assert_eq!(
        InternalPll::CNN_FREQUENCY,
        2 * <InternalPll as crate::gcr::clocks::OscillatorSource>::BASE_FREQUENCY
    );
}

#[test]
fn full_speed_needs_the_pipeline_and_the_pll() {
    // Only the PLL undivided reaches the rated maximum.
    assert_eq!(
        InternalPll::CNN_FREQUENCY / CnnClockDiv::Div1.divisor(),
        MAX_PIPELINED_FREQUENCY
    );
    // Div4 is the fastest PLL setting a non-pipelined part can take.
    assert!(
        InternalPll::CNN_FREQUENCY / CnnClockDiv::Div4.divisor() <= MAX_NON_PIPELINED_FREQUENCY
    );
    assert!(InternalPll::CNN_FREQUENCY / CnnClockDiv::Div2.divisor() > MAX_NON_PIPELINED_FREQUENCY);
}

// ---------------------------------------------------------------- control words
/// Init constants, cross-checked against the `cnn_init` of every shipped
/// example. `kws20_demo` has no bias and writes `0x1880`; the other seven
/// have bias and write `0x1c80`.
#[test]
fn init_constants_match_the_generated_sources() {
    assert_eq!(SRAM_CONTROL, 0x0000_040e);
    assert_eq!(ZEROIZE_NO_BIAS, 0x0000_1880);
    assert_eq!(ZEROIZE_WITH_BIAS, 0x0000_1c80);
    // Bias selection is the only difference between the two.
    assert_eq!(ZEROIZE_WITH_BIAS ^ ZEROIZE_NO_BIAS, 1 << 10);
    // Both run the zeroize, and neither runs any BIST.
    for word in [ZEROIZE_NO_BIAS, ZEROIZE_WITH_BIAS] {
        assert_ne!(word & (1 << 7), 0, "ZERO_RUN clear");
        assert_eq!(word & 0b101_0101, 0, "a BIST run bit is set");
    }
}

/// The arm-and-go sequence, cross-checked against `cnn_start` in
/// `kws20_demo`. The master arms with `CNN_EN` clear and the others with it
/// set; inverting that hangs the accelerator. Both words keep the APB clock
/// alive (bit 3) and use memory-express weight loading (bit 20).
#[test]
fn start_sequence_matches_the_generated_sources() {
    assert_eq!(STOP_SM, 0x0010_0008);
    assert_eq!(START_MASTER, 0x0010_0808);
    assert_eq!(START_OTHER, 0x0010_0809);
    assert_eq!(START_GO, 0x0010_0009);

    // Bits 10:9 carry the master quadrant index.
    assert_eq!((START_MASTER >> 9) & 0b11, MASTER_QUADRANT as u32);
    assert_eq!(START_MASTER & 1, 0, "master must arm with CNN_EN clear");
    assert_eq!(START_MASTER | 1, START_OTHER, "they differ only in CNN_EN");
    for word in [STOP_SM, START_MASTER, START_OTHER, START_GO] {
        assert_ne!(word & (1 << 3), 0, "{word:#010x} has CLK_EN clear");
        assert_ne!(word & (1 << 20), 0, "{word:#010x} has MEXPRESS clear");
    }
}

/// The generator folds `NO_PIPELINE` into every control word it builds, so
/// it cannot be written once at init and left alone.
#[test]
fn pipeline_contributes_to_every_control_word() {
    assert_eq!(Pipeline::Enabled.ctl_bits(), 0);
    assert_eq!(Pipeline::Disabled.ctl_bits(), 1 << 5);

    let p = Pipeline::Disabled.ctl_bits();
    for word in [STOP_SM, START_MASTER, START_OTHER, START_GO] {
        assert_eq!(word & (1 << 5), 0, "{word:#010x} already has NO_PIPELINE");
    }
    assert_eq!(STOP_SM | p, 0x0010_0028);
    assert_eq!(START_MASTER | p, 0x0010_0828);
    assert_eq!(START_GO | p, 0x0010_0029);
}

/// The two acknowledge masks the generator emits. A third form exists for
/// one-shot mode, which this HAL does not expose.
#[test]
fn acknowledge_masks_match_the_generated_isr() {
    // kws20_demo: `&= ~((1 << 12) | 1)`
    // A streaming network also clears STREAM_EN, bit 14, which this HAL
    // never sets.
    assert_eq!(ack_mask(), (1 << 12) | 1);
    // Clearing DONE is what makes the next completion detectable.
    assert_ne!(ack_mask() & (1 << 12), 0);
}

/// `stop` and `resume` toggle the same bit the go word sets, so a stopped
/// network resumes exactly where the go left it.
#[test]
fn stop_and_resume_toggle_the_enable_bit() {
    assert_eq!(START_GO & 1, 1);
    assert_ne!(ack_mask() & 1, 0, "the ISR also clears CNN_EN");
}

/// `LCNT_MAX` takes hardware layer indices, so the last index is one less
/// than the layer count. Values from the shipped examples.
#[test]
fn layer_count_encoding() {
    // kws20_demo: 9 layers, imagenet: 34, cifar-100-effnet2: 33,
    // mobilefacenet-112: 73.
    for (layers, expected) in [(9u32, 0x08u32), (34, 0x21), (33, 0x20), (73, 0x48)] {
        let last = layers - 1;
        assert_eq!(last | (0 << 8), expected, "{layers} layers");
    }
}

// ---------------------------------------------------------------- boost
#[derive(Default)]
struct FakePin {
    high: bool,
}

impl ErrorType for FakePin {
    type Error = core::convert::Infallible;
}

impl OutputPin for FakePin {
    fn set_high(&mut self) -> Result<(), Self::Error> {
        self.high = true;
        Ok(())
    }

    fn set_low(&mut self) -> Result<(), Self::Error> {
        self.high = false;
        Ok(())
    }
}

/// The C helper drives the pin high to enable, which is the default here.
#[test]
fn active_high_drives_high_to_enable() {
    let mut boost = CnnBoost::new(FakePin::default());
    boost.enable().unwrap();
    assert!(boost.release().high);

    let mut boost = CnnBoost::new(FakePin::default());
    boost.disable().unwrap();
    assert!(!boost.release().high);
}

/// A load switch with an inverting enable is a board decision, not a
/// property of the accelerator, so both polarities have to work.
#[test]
fn active_low_inverts_both_directions() {
    let mut boost = CnnBoost::with_polarity(FakePin::default(), BoostPolarity::ActiveLow);
    boost.enable().unwrap();
    assert!(!boost.release().high);

    let mut boost = CnnBoost::with_polarity(FakePin::default(), BoostPolarity::ActiveLow);
    boost.disable().unwrap();
    assert!(boost.release().high);
}

// ---------------------------------------------------------------- golden corpus
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
            let net: Network = Network::new(&layers, 0, 0, &[], None, &[], &[]);
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
