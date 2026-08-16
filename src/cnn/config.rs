//! Layer Configuration
//!
//! Turns a Layer into the sequence of register writes that programs it.
//!

use super::fields::LayerRegister;
use super::network::{InputMode, Layer, Network};
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
    pub fn configure<M: InputMode>(&mut self, network: &Network<M>) {
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
    use crate::cnn::network::{Direct, Stream};
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

        fn writes(&self) -> &[(u32, u32)] {
            &self.writes[..self.len]
        }
    }

    fn blank() -> Layer {
        Layer {
            next: Nxtlyr::new(),
            rows: Rcnt::new(),
            cols: Ccnt::new(),
            oned: Oned::new(),
            pool_rows: Prcnt::new(),
            pool_cols: Pccnt::new(),
            stride: Stride::new(),
            wptr_ts: WptrToffs::new(),
            wptr_moffs: None,
            wptr_choffs: WptrChoffs::new(),
            rptr: RptrBase::new(),
            lctl2: Lctl2::new(),
            mcnt1: None,
            mcnt2: Mcnt2::new(),
            ochan: Ochan::new(),
            tptr: Tptr::new(),
            lctl: [Lctl::new(); 4],
            post: [Post::new(); 4],
            wptr: [WptrBase::new(); 4],
            ena: [Ena::new(); 4],
            stream: None,
            master_only: false,
        }
    }

    /// kws20_demo layer 0, transcribed from the generated `cnn.c`.
    fn kws20_layer0() -> Layer {
        Layer {
            rows: Rcnt::from_bits(0x0002_007f),
            cols: Ccnt::from_bits(0x0001_0000),
            stride: Stride::from_bits(0x0000_0020),
            wptr_moffs: Some(WptrMoffs::from_bits(0x0000_8000)),
            wptr_choffs: WptrChoffs::from_bits(0x0000_0001),
            lctl2: Lctl2::from_bits(0x0019_8011),
            mcnt1: Some(Mcnt1::from_bits(0x0000_0678)),
            ochan: Ochan::from_bits(0x0000_00cf),
            oned: Oned::from_bits(0x0000_1100),
            lctl: [
                Lctl::from_bits(0x0000_eb20),
                Lctl::from_bits(0x0000_0b20),
                Lctl::from_bits(0x0000_0b20),
                Lctl::from_bits(0x0000_0b20),
            ],
            post: [Post::from_bits(0x0000_2000); 4],
            wptr: [WptrBase::from_bits(0x0000_0800); 4],
            ena: [Ena::from_bits(0xffff_ffff); 4],
            ..blank()
        }
    }

    /// The whole point of H10: the emitted sequence, offsets and values, must
    /// match the generated source exactly. This is `// Layer 0 quadrant 0` of
    /// `kws20_demo`, in order.
    #[test]
    fn kws20_layer0_quadrant0_matches_the_generated_source() {
        assert_eq!(
            Recorder::of(&kws20_layer0(), 0).writes(),
            [
                (0x04, 0x0002_007f), // Rows
                (0x08, 0x0001_0000), // Columns
                (0x18, 0x0000_0020), // Stride
                (0x1c, 0x0000_0800), // SRAM write ptr
                (0x24, 0x0000_8000), // Write ptr mask offs
                (0x28, 0x0000_0001), // Write ptr multi-pass channel offs
                (0x30, 0x0000_eb20), // Layer control
                (0x34, 0x0019_8011), // Layer control 2
                (0x38, 0x0000_0678), // Mask count
                (0x40, 0x0000_00cf), // Output channel count
                (0x0c, 0x0000_1100), // 1D
                (0x4c, 0x0000_2000), // Post processing register
                (0x48, 0xffff_ffff), // Mask and processor enables
            ]
        );
    }

    /// The same layer on a non-master quadrant differs only in `LCTL`, because
    /// `SIENA` is set on the master alone.
    #[test]
    fn non_master_quadrant_differs_only_in_layer_control() {
        let layer = kws20_layer0();
        let master = Recorder::of(&layer, 0);
        let other = Recorder::of(&layer, 1);
        assert_eq!(master.len, other.len);

        for (a, b) in master.writes().iter().zip(other.writes()) {
            assert_eq!(a.0, b.0, "emit order differs between quadrants");
            if a.0 == LayerReg::Lctl as u32 {
                assert_eq!((a.1, b.1), (0x0000_eb20, 0x0000_0b20));
            } else {
                assert_eq!(a, b, "register {:#04x} differs", a.0);
            }
        }
    }

    /// Thirteen of the twenty registers are written; the other seven are zero
    /// and must not appear at all.
    #[test]
    fn zero_valued_registers_are_suppressed() {
        let writes = Recorder::of(&kws20_layer0(), 0);
        assert_eq!(writes.len, 13);
        for (_, bits) in writes.writes() {
            assert_ne!(*bits, 0);
        }
        for skipped in [
            LayerReg::Next,
            LayerReg::PoolRows,
            LayerReg::PoolCols,
            LayerReg::WptrTs,
            LayerReg::Rptr,
            LayerReg::Moffs,
            LayerReg::Tptr,
        ] {
            assert!(
                !writes.writes().iter().any(|(r, _)| *r == skipped as u32),
                "{skipped:?} should have been suppressed"
            );
        }

        // A layer with nothing set emits nothing.
        assert_eq!(Recorder::of(&blank(), 0).len, 0);
    }

    /// The emitted order must be the declared emit order, restricted to the
    /// registers that were written.
    #[test]
    fn emit_follows_the_declared_order() {
        let mut full = blank();
        // Give every register a non-zero value so none is suppressed.
        full.next = Nxtlyr::from_bits(1);
        full.rows = Rcnt::from_bits(1);
        full.cols = Ccnt::from_bits(1);
        full.oned = Oned::from_bits(1);
        full.pool_rows = Prcnt::from_bits(1);
        full.pool_cols = Pccnt::from_bits(1);
        full.stride = Stride::from_bits(1);
        full.wptr_ts = WptrToffs::from_bits(1);
        full.wptr_moffs = Some(WptrMoffs::from_bits(1));
        full.wptr_choffs = WptrChoffs::from_bits(1);
        full.rptr = RptrBase::from_bits(1);
        full.lctl2 = Lctl2::from_bits(1);
        full.mcnt1 = Some(Mcnt1::from_bits(1));
        full.mcnt2 = Mcnt2::from_bits(1);
        full.ochan = Ochan::from_bits(1);
        full.tptr = Tptr::from_bits(1);
        full.lctl = [Lctl::from_bits(1); 4];
        full.post = [Post::from_bits(1); 4];
        full.wptr = [WptrBase::from_bits(1); 4];
        full.ena = [Ena::from_bits(1); 4];

        let emitted = Recorder::of(&full, 0);
        assert_eq!(emitted.len, 20, "every register should have been written");

        let expected: [u32; 20] = core::array::from_fn(|i| crate::cnn::EMIT_ORDER[i] as u32);
        let actual: [u32; 20] = core::array::from_fn(|i| emitted.writes()[i].0);
        assert_eq!(actual, expected);
    }

    /// Passthrough layers leave these two registers at whatever init wrote,
    /// which is zero. `Some(0)` would be indistinguishable from `None` after
    /// suppression, so absence is what carries the meaning.
    #[test]
    fn passthrough_skips_two_registers() {
        let mut layer = kws20_layer0();
        layer.wptr_moffs = None;
        layer.mcnt1 = None;

        let writes = Recorder::of(&layer, 0);
        assert_eq!(writes.len, 11);
        assert!(!writes
            .writes()
            .iter()
            .any(|(r, _)| *r == LayerReg::WptrMask as u32));
        assert!(!writes
            .writes()
            .iter()
            .any(|(r, _)| *r == LayerReg::Mcnt as u32));
    }

    /// Streaming registers live outside the per-layer file, so they are not
    /// part of this sequence.
    #[test]
    fn streaming_is_not_emitted_here() {
        let mut layer = kws20_layer0();
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
        assert_eq!((Direct::START_MASTER >> 9) & 0b11, MASTER_QUADRANT as u32);
    }
}
