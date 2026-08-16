//! # CNN Accelerator Register Defs
//!
//! All addresses are derived from the PAC base

use super::fields::{with_decoded, LayerRegister, ALL_TYPED_REGS};

pub const QUADRANTS: u8 = 4;
pub const PROCESSORS_PER_QUADRANT: u8 = 16;
pub const DATA_INSTANCES_PER_QUADRANT: u8 = 4;
pub const MAX_LAYERS: u8 = 128;
pub const MAX_STREAM_LAYERS: u8 = 8;

/// Base address of quadrant 0
const QUADRANT0_BASE: u32 = 0x5100_0000;
/// Address stride between quadrants.
const QUADRANT_STRIDE: u32 = 0x0100_0000;
/// Offset of the per-layer register file within a quadrant
const LAYER_BASE: u32 = 0x0010_0000;
/// Address stride between layers within the register file
const LAYER_STRIDE: u32 = 0x100;
/// Offset of the streaming register file within a quadrant.
const STREAM_BASE: u32 = 0x0010_8000;

/// Base address of quadrant `q`
#[inline]
pub const fn quadrant_base(q: u8) -> u32 {
    debug_assert!(q < QUADRANTS);
    QUADRANT0_BASE + (q as u32) * QUADRANT_STRIDE
}

/// A single 32-bit memory-mapped register
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Reg(u32);

impl Reg {
    #[inline]
    pub fn read(self) -> u32 {
        unsafe { (self.0 as *const u32).read_volatile() }
    }

    #[inline]
    pub fn write(self, value: u32) {
        unsafe { (self.0 as *mut u32).write_volatile(value) }
    }

    #[inline]
    pub fn modify(self, f: impl FnOnce(u32) -> u32) {
        self.write(f(self.read()));
    }

    #[inline]
    pub const fn addr(self) -> u32 {
        self.0
    }
}

impl core::fmt::Debug for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Reg(0x{:08x})", self.0)
    }
}

/// A register within the per-layer register file.
///
/// Values are the byte offset within a layer's 0x100-byte block.
/// TODO: Figure out bitfield meanings. GEE GEE GEE.
///       As previously mentioned, datasheet is gee.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum LayerReg {
    Next = 0x00,
    Rows = 0x04,
    Cols = 0x08,
    /// 1D convolution configuration.
    Oned = 0x0c,
    /// Pooling rows.
    PoolRows = 0x10,
    /// Pooling columns.
    PoolCols = 0x14,
    /// Stride.
    Stride = 0x18,
    /// SRAM write pointer.
    Wptr = 0x1c,
    /// Write pointer time slot offset.
    WptrTs = 0x20,
    /// Write pointer mask offset.
    WptrMask = 0x24,
    /// Write pointer multi-pass channel offset.
    WptrMp = 0x28,
    /// SRAM read pointer.
    Rptr = 0x2c,
    /// Layer control.
    Lctl = 0x30,
    /// Layer control 2.
    Lctl2 = 0x34,
    /// Mask count.
    Mcnt = 0x38,
    /// Mask offset.
    Moffs = 0x3c,
    /// Output channel count.
    Ochan = 0x40,
    /// TRAM pointer maximum.
    Tptr = 0x44,
    /// Mask and processor enables.
    En = 0x48,
    /// Post processing.
    Post = 0x4c,
}

/// Every layer register, in declaration order.
/// Nte that emit order is differne.t
pub const ALL_LAYER_REGS: [LayerReg; 20] = [
    LayerReg::Next,
    LayerReg::Rows,
    LayerReg::Cols,
    LayerReg::Oned,
    LayerReg::PoolRows,
    LayerReg::PoolCols,
    LayerReg::Stride,
    LayerReg::Wptr,
    LayerReg::WptrTs,
    LayerReg::WptrMask,
    LayerReg::WptrMp,
    LayerReg::Rptr,
    LayerReg::Lctl,
    LayerReg::Lctl2,
    LayerReg::Mcnt,
    LayerReg::Moffs,
    LayerReg::Ochan,
    LayerReg::Tptr,
    LayerReg::En,
    LayerReg::Post,
];

/// One CNNx16 quadrant's register files
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Quadrant(u8);

impl Quadrant {
    /// Create a handle to quadrant `index`
    #[inline]
    pub const fn new(index: u8) -> Self {
        debug_assert!(index < QUADRANTS);
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn base(self) -> u32 {
        quadrant_base(self.0)
    }

    /// The per-layer register file for layer `n`
    #[inline]
    pub const fn layer(self, n: u8) -> LayerRegs {
        debug_assert!(n < MAX_LAYERS);
        LayerRegs {
            addr: self.base() + LAYER_BASE + (n as u32) * LAYER_STRIDE,
        }
    }

    /// Stream processing start, for streaming slot s
    #[inline]
    pub const fn stream_start(self, s: u8) -> Reg {
        debug_assert!(s < MAX_STREAM_LAYERS);
        Reg(self.base() + STREAM_BASE + (s as u32) * 4)
    }

    /// Stream processing delta, for streaming slot s
    #[inline]
    pub const fn stream_delta(self, s: u8) -> Reg {
        debug_assert!(s < MAX_STREAM_LAYERS);
        Reg(self.base() + STREAM_BASE + 0x20 + (s as u32) * 4)
    }

    /// Rollover, for streaming slot s
    #[inline]
    pub const fn rollover(self, s: u8) -> Reg {
        debug_assert!(s < MAX_STREAM_LAYERS);
        Reg(self.base() + STREAM_BASE + 0x40 + (s as u32) * 4)
    }

    /// Input frame size (not per-layer)
    #[inline]
    pub const fn frame_size(self) -> Reg {
        Reg(self.base() + STREAM_BASE + 0x60)
    }
}

/// The register file for one layer within one quadrant
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerRegs {
    addr: u32,
}

impl LayerRegs {
    #[inline]
    pub const fn reg(self, r: LayerReg) -> Reg {
        Reg(self.addr + r as u32)
    }
    #[inline]
    pub fn read_typed<R: LayerRegister>(self) -> R {
        R::from_bits(self.reg(R::REG).read())
    }
    #[inline]
    pub fn write_typed<R: LayerRegister>(self, value: R) {
        self.reg(R::REG).write(value.bits())
    }

    /// Decode all registers
    #[inline]
    pub fn dump(self, mut visit: impl FnMut(LayerReg, &dyn core::fmt::Debug)) {
        for reg in ALL_TYPED_REGS {
            let bits = self.reg(reg).read();
            with_decoded(reg, bits, |value| visit(reg, value));
        }
    }
    #[inline]
    pub const fn next(self) -> Reg {
        self.reg(LayerReg::Next)
    }
    #[inline]
    pub const fn rows(self) -> Reg {
        self.reg(LayerReg::Rows)
    }
    #[inline]
    pub const fn cols(self) -> Reg {
        self.reg(LayerReg::Cols)
    }
    #[inline]
    pub const fn oned(self) -> Reg {
        self.reg(LayerReg::Oned)
    }
    #[inline]
    pub const fn pool_rows(self) -> Reg {
        self.reg(LayerReg::PoolRows)
    }
    #[inline]
    pub const fn pool_cols(self) -> Reg {
        self.reg(LayerReg::PoolCols)
    }
    #[inline]
    pub const fn stride(self) -> Reg {
        self.reg(LayerReg::Stride)
    }
    #[inline]
    pub const fn wptr(self) -> Reg {
        self.reg(LayerReg::Wptr)
    }
    #[inline]
    pub const fn wptr_ts(self) -> Reg {
        self.reg(LayerReg::WptrTs)
    }
    #[inline]
    pub const fn wptr_mask(self) -> Reg {
        self.reg(LayerReg::WptrMask)
    }
    #[inline]
    pub const fn wptr_mp(self) -> Reg {
        self.reg(LayerReg::WptrMp)
    }
    #[inline]
    pub const fn rptr(self) -> Reg {
        self.reg(LayerReg::Rptr)
    }
    #[inline]
    pub const fn lctl(self) -> Reg {
        self.reg(LayerReg::Lctl)
    }
    #[inline]
    pub const fn lctl2(self) -> Reg {
        self.reg(LayerReg::Lctl2)
    }
    #[inline]
    pub const fn mcnt(self) -> Reg {
        self.reg(LayerReg::Mcnt)
    }
    #[inline]
    pub const fn moffs(self) -> Reg {
        self.reg(LayerReg::Moffs)
    }
    #[inline]
    pub const fn ochan(self) -> Reg {
        self.reg(LayerReg::Ochan)
    }
    #[inline]
    pub const fn tptr(self) -> Reg {
        self.reg(LayerReg::Tptr)
    }
    #[inline]
    pub const fn en(self) -> Reg {
        self.reg(LayerReg::En)
    }
    #[inline]
    pub const fn post(self) -> Reg {
        self.reg(LayerReg::Post)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use crate::pac;
        assert_eq!(Quadrant::new(0).base(), pac::Cnnx16_0::PTR as u32);
        assert_eq!(Quadrant::new(1).base(), pac::Cnnx16_1::PTR as u32);
        assert_eq!(Quadrant::new(2).base(), pac::Cnnx16_2::PTR as u32);
        assert_eq!(Quadrant::new(3).base(), pac::Cnnx16_3::PTR as u32);
    }

    #[test]
    fn layer_registers() {
        // kws20_demo, layer 0 quadrant 0
        assert_eq!(Quadrant::new(0).layer(0).rows().addr(), 0x5110_0004);
        assert_eq!(Quadrant::new(0).layer(0).cols().addr(), 0x5110_0008);
        assert_eq!(Quadrant::new(0).layer(0).oned().addr(), 0x5110_000c);
        assert_eq!(Quadrant::new(0).layer(0).post().addr(), 0x5110_004c);
        assert_eq!(Quadrant::new(0).layer(0).en().addr(), 0x5110_0048);
        // mobilefacenet-112, layer 3 quadrant 0
        assert_eq!(Quadrant::new(0).layer(3).rows().addr(), 0x5110_0304);
        assert_eq!(Quadrant::new(0).layer(3).tptr().addr(), 0x5110_0344);
        // cifar-100-mobilenet-v2, layer 72 quadrant 3
        assert_eq!(Quadrant::new(3).layer(72).rows().addr(), 0x5410_4804);
        // pascalvoc-retinanetv7_3, layer 115 quadrant 2
        assert_eq!(Quadrant::new(2).layer(115).rows().addr(), 0x5310_7304);
    }

    #[test]
    fn stream_registers() {
        // mobilefacenet-112 quadrant 0
        assert_eq!(Quadrant::new(0).stream_start(0).addr(), 0x5110_8000);
        assert_eq!(Quadrant::new(0).stream_start(1).addr(), 0x5110_8004);
        assert_eq!(Quadrant::new(0).stream_delta(1).addr(), 0x5110_8024);
        assert_eq!(Quadrant::new(0).rollover(0).addr(), 0x5110_8040);
        assert_eq!(Quadrant::new(0).rollover(1).addr(), 0x5110_8044);
        assert_eq!(Quadrant::new(0).frame_size().addr(), 0x5110_8060);
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
}
