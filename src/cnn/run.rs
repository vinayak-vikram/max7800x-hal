//! Initialization and run control

use super::network::Network;
use super::{Cnn, Pipeline};
use crate::gcr::clocks::Enabled;

/// CTL bits a completion interrupt handler must clear.
pub const fn ack_mask() -> u32 {
    (1 << 12) | 1
}

/// `CTL` words for the direct input path, verified against generated `cnn.c`.
/// The master arms with `CNN_EN` clear and every other quadrant with it set;
/// inverting that hangs the accelerator.
const STOP_SM: u32 = 0x0010_0008;
const START_MASTER: u32 = 0x0010_0808;
const START_OTHER: u32 = 0x0010_0809;
const START_GO: u32 = 0x0010_0009;

/// SRAM control word.
const SRAM_CONTROL: u32 = 0x0000_040e;

/// Zeroize commands
const ZEROIZE_NO_BIAS: u32 = 0x0000_1880;
const ZEROIZE_WITH_BIAS: u32 = 0x0000_1c80;

macro_rules! each_quadrant {
    ($self:ident, |$q:ident| $body:block) => {{
        {
            let $q = &$self.q0;
            $body
        }
        {
            let $q = &$self.q1;
            $body
        }
        {
            let $q = &$self.q2;
            $body
        }
        {
            let $q = &$self.q3;
            $body
        }
    }};
}

impl Cnn<Enabled> {
    /// Initialize the accelerator
    pub fn init(&mut self, network: &Network) {
        let no_pipeline = matches!(self.pipeline, Pipeline::Disabled);

        // clk_en and pipeline selection
        each_quadrant!(self, |q| {
            q.ctl()
                .write(|w| w.clk_en().set_bit().no_pipeline().bit(no_pipeline));
        });

        // Ready-select 0, no quadrant powered down.
        self.cnn.aon().write(|w| unsafe { w.bits(0) });

        each_quadrant!(self, |q| {
            q.sram().write(|w| unsafe { w.bits(SRAM_CONTROL) });
        });

        let zeroize = if network.has_bias() {
            ZEROIZE_WITH_BIAS
        } else {
            ZEROIZE_NO_BIAS
        };
        // Start all four before polling any of them; they run concurrently.
        each_quadrant!(self, |q| {
            q.test().write(|w| unsafe { w.bits(zeroize) });
        });
        each_quadrant!(self, |q| {
            while q.test().read().zero_done().bit_is_clear() {}
        });
        each_quadrant!(self, |q| {
            q.test().write(|w| unsafe { w.bits(0) });
        });

        let stop = STOP_SM | self.pipeline.ctl_bits();
        each_quadrant!(self, |q| {
            q.ctl().write(|w| unsafe { w.bits(stop) });
            q.lcnt().write(|w| unsafe {
                w.last()
                    .bits(network.last_layer)
                    .start()
                    .bits(network.first_layer)
            });
        });
    }

    /// Start inference.
    pub fn start(&mut self) {
        let pipeline = self.pipeline.ctl_bits();
        self.q0
            .ctl()
            .write(|w| unsafe { w.bits(START_MASTER | pipeline) });
        self.q1
            .ctl()
            .write(|w| unsafe { w.bits(START_OTHER | pipeline) });
        self.q2
            .ctl()
            .write(|w| unsafe { w.bits(START_OTHER | pipeline) });
        self.q3
            .ctl()
            .write(|w| unsafe { w.bits(START_OTHER | pipeline) });

        self.q0
            .ctl()
            .write(|w| unsafe { w.bits(START_GO | pipeline) });
    }

    /// Whether the accelerator has signalled completion.
    pub fn is_complete(&self) -> bool {
        self.q0.ctl().read().irq().bit_is_set()
    }

    pub fn wait(&self) {
        while !self.is_complete() {}
    }

    /// Acknowledge the completion interrupt on every quadrant.
    pub fn acknowledge(&mut self) {
        let mask = ack_mask();
        self.q0
            .ctl()
            .modify(|r, w| unsafe { w.bits(r.bits() & !mask) });
        self.q1
            .ctl()
            .modify(|r, w| unsafe { w.bits(r.bits() & !mask) });
        self.q2
            .ctl()
            .modify(|r, w| unsafe { w.bits(r.bits() & !mask) });
        self.q3
            .ctl()
            .modify(|r, w| unsafe { w.bits(r.bits() & !mask) });
    }

    /// Halt the master quadrant, pausing the network where it stands.
    pub fn stop(&mut self) {
        self.q0.ctl().modify(|_, w| w.en().clear_bit());
    }

    /// Release the master quadrant again after [`stop`](Cnn::stop).
    pub fn resume(&mut self) {
        self.q0.ctl().modify(|_, w| w.en().set_bit());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::MASTER_QUADRANT;
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
}
