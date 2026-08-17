//! # CNN Accelerator Register Defs
//!
//! All addresses are derived from the PAC base

use super::fields::{with_decoded, ALL_TYPED_REGS};

pub const QUADRANTS: u8 = 4;
pub const PROCESSORS_PER_QUADRANT: u8 = 16;
pub const DATA_INSTANCES_PER_QUADRANT: u8 = 4;
pub const MAX_LAYERS: u8 = 128;

/// Base address of quadrant 0
const QUADRANT0_BASE: u32 = 0x5100_0000;
/// Address stride between quadrants.
const QUADRANT_STRIDE: u32 = 0x0100_0000;
/// Offset of the per-layer register file within a quadrant
const LAYER_BASE: u32 = 0x0010_0000;
/// Address stride between layers within the register file
const LAYER_STRIDE: u32 = 0x100;

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
/// Values are the byte offset within a layer's 0x100-byte block. See
/// [`fields`](super::fields) for the bitfields at each one.
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
    /// Decode all registers
    #[inline]
    pub fn dump(self, mut visit: impl FnMut(LayerReg, &dyn core::fmt::Debug)) {
        for reg in ALL_TYPED_REGS {
            let bits = self.reg(reg).read();
            with_decoded(reg, bits, |value| visit(reg, value));
        }
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
}
