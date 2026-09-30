//! # CNN Accelerator
//!
//! The datasheet is gee.
//!
//! Four CNNx16 quadrants, 16 processors each; 32 layers on the MAX78000 and
//! 128 on the MAX78002. A [`Network`] is a `const` descriptor produced by
//! `tools/cnn-gen.py` for one of the two; nothing here computes a register
//! value.
//!
//! Call order, matching the generated C:
//!
//! ```text
//! enable -> init -> load_weights -> load_bias -> configure -> write_u32
//!        -> start -> wait -> read_u32
//! ```
//!
//! Direct input only. FIFO input and streaming layers are not implemented; see
//! the note at the top of [`network`] if they are ever wanted back.
//!
//! [`regs::Reg::write`] is one `write_volatile` at a computed address, the same
//! mechanism as the SDK's raw C pokes with types above it.

pub mod boost;
pub mod config;
pub mod data;
pub mod fields;
pub mod memory;
pub mod network;
pub mod power;
pub mod regs;
pub mod run;
pub mod validate;

#[cfg(all(test, feature = "max78002"))]
mod tests;
#[cfg(all(test, feature = "max78000"))]
mod tests_max78000;

pub use boost::{BoostPolarity, CnnBoost};
pub use config::{emit_layer, LayerSink, MASTER_QUADRANT};
pub use network::{InputRegion, Layer, Network, OutputRegion, WeightRegion};
pub use power::{CnnClockDiv, CnnClockSource, LOAD_SWITCH_SETTLE_MS};
#[cfg(feature = "max78002")]
pub use power::{Pipeline, MAX_NON_PIPELINED_FREQUENCY, MAX_PIPELINED_FREQUENCY};
pub use regs::{LayerReg, LayerRegs, Quadrant, Reg};
pub use run::ack_mask;
pub use validate::Invalid;

use crate::gcr::clocks::{Disabled, Enabled};
use core::marker::PhantomData;

#[doc(hidden)]
pub trait CnnState: crate::Sealed {}
impl CnnState for Disabled {}
impl CnnState for Enabled {}

/// CNN accelerator
pub struct Cnn<S: CnnState> {
    cnn: crate::pac::Cnn,
    q0: crate::pac::Cnnx16_0,
    q1: crate::pac::Cnnx16_1,
    q2: crate::pac::Cnnx16_2,
    q3: crate::pac::Cnnx16_3,
    gcfr: crate::pac::Gcfr,
    #[cfg(feature = "max78002")]
    pipeline: Pipeline,
    source: Option<CnnClockSource>,
    divider: CnnClockDiv,
    _state: PhantomData<S>,
}

impl<S: CnnState> Cnn<S> {
    /// The configured pipeline mode
    #[cfg(feature = "max78002")]
    pub const fn pipeline(&self) -> Pipeline {
        self.pipeline
    }

    /// Register file for one quadrant
    pub const fn quadrant(&self, index: u8) -> Quadrant {
        Quadrant::new(index)
    }
}
