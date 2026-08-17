//! Layer Configuration
//!
//! Turns a Layer into the sequence of register writes that programs it.
//!

use super::fields::LayerRegister;
use super::network::{Layer, Network};
use super::regs::{LayerRegs, Quadrant, QUADRANTS};

/// Index of the master quadrant.
pub const MASTER_QUADRANT: u8 = 0;

/// Somewhere register writes go. Trait exists for debug reasons and to compare against C impls
pub trait LayerSink {
    fn write(&mut self, reg: super::LayerReg, bits: u32);
}

impl LayerSink for LayerRegs {
    #[inline]
    fn write(&mut self, reg: super::LayerReg, bits: u32) {
        self.reg(reg).write(bits);
    }
}

/// Writes a register unless its value is zero.
#[inline]
fn emit<R: LayerRegister>(sink: &mut impl LayerSink, value: R) {
    let bits = value.bits();
    if bits != 0 {
        sink.write(R::REG, bits);
    }
}

/// Write one layer's configuration for one quadrant.
/// This order is the generator's, not the address order. `Oned` follows
/// `Ochan`, and `Ena` is last because it arms the processors.
pub fn emit_layer(sink: &mut impl LayerSink, layer: &Layer, quadrant: u8) {
    let q = quadrant as usize;

    emit(sink, layer.next);
    emit(sink, layer.rows);
    emit(sink, layer.cols);
    emit(sink, layer.pool_rows);
    emit(sink, layer.pool_cols);
    emit(sink, layer.stride);
    emit(sink, layer.wptr[q]);
    emit(sink, layer.wptr_ts);
    if let Some(moffs) = layer.wptr_moffs {
        emit(sink, moffs);
    }
    emit(sink, layer.wptr_choffs);
    emit(sink, layer.rptr);
    emit(sink, layer.lctl[q]);
    emit(sink, layer.lctl2);
    if let Some(mcnt1) = layer.mcnt1 {
        emit(sink, mcnt1);
    }
    emit(sink, layer.mcnt2);
    emit(sink, layer.ochan);
    emit(sink, layer.oned);
    emit(sink, layer.tptr);
    emit(sink, layer.post[q]);
    emit(sink, layer.ena[q]);
}

impl super::Cnn<crate::gcr::clocks::Enabled> {
    /// Write every layer of `network` into the register file.
    pub fn configure(&mut self, network: &Network) {
        for (index, layer) in network.layers.iter().enumerate() {
            debug_assert!(
                !layer.is_synthetic() || layer.master_only,
                "an inserted average-pool reset layer must be master-only"
            );
            for quadrant in 0..QUADRANTS {
                if layer.master_only && quadrant != MASTER_QUADRANT {
                    continue;
                }
                let mut regs = Quadrant::new(quadrant).layer(index as u8);
                emit_layer(&mut regs, layer, quadrant);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::fields::*;
    use crate::cnn::network::Stream;
    use crate::cnn::LayerReg;

    /// Records what a layer emits for debug reasons.
    #[derive(Default)]
    struct Recorder {
        writes: [(u32, u32); 24],
        len: usize,
    }

    impl LayerSink for Recorder {
        fn write(&mut self, reg: LayerReg, bits: u32) {
            self.writes[self.len] = (reg as u32, bits);
            self.len += 1;
        }
    }

    impl Recorder {
        fn of(layer: &Layer, quadrant: u8) -> Self {
            let mut r = Self::default();
            emit_layer(&mut r, layer, quadrant);
            r
        }
    }

    /// Streaming registers live outside the per-layer file, so they are not
    /// part of this sequence.
    #[test]
    fn streaming_is_not_emitted_here() {
        let mut layer = Layer {
            rows: Rcnt::from_bits(0x0002_007f),
            ..Layer::default()
        };
        let without = Recorder::of(&layer, 0).len;
        layer.stream = Some(Stream {
            slot: 0,
            start: Stream1::from_bits(0x147),
            delta: Stream2::from_bits(0x0142_0022),
            rollover: Fmax::from_bits(0x148),
        });
        assert_eq!(Recorder::of(&layer, 0).len, without);
    }

    #[test]
    fn master_quadrant_is_zero() {
        assert_eq!(MASTER_QUADRANT, 0);
        // The `CTL` arm words encode the master index in bits 10:9.
        assert_eq!(
            (super::super::START_MASTER >> 9) & 0b11,
            MASTER_QUADRANT as u32
        );
    }
}
