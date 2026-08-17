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
pub(super) const STOP_SM: u32 = 0x0010_0008;
pub(super) const START_MASTER: u32 = 0x0010_0808;
pub(super) const START_OTHER: u32 = 0x0010_0809;
pub(super) const START_GO: u32 = 0x0010_0009;

/// SRAM control word.
pub(super) const SRAM_CONTROL: u32 = 0x0000_040e;

/// Zeroize commands
pub(super) const ZEROIZE_NO_BIAS: u32 = 0x0000_1880;
pub(super) const ZEROIZE_WITH_BIAS: u32 = 0x0000_1c80;

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
