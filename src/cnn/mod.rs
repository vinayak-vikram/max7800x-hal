//! # CNN Accelerator (impl for MAX78002)
//!
//! The datasheet is gee.
//!

pub mod memory;
pub mod regs;

pub use regs::{LayerReg, LayerRegs, Quadrant, Reg};

/// The order in which a layer's registers must be written.
pub const EMIT_ORDER: [LayerReg; 20] = [
    LayerReg::Next,
    LayerReg::Rows,
    LayerReg::Cols,
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
    LayerReg::Oned,
    LayerReg::Tptr,
    LayerReg::Post,
    LayerReg::En,
];
