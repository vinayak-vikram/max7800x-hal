//! # CNN Accelerator (impl for MAX78002)
//!
//! The datasheet is gee.
//!

pub mod fields;
pub mod memory;
pub mod network;
pub mod regs;

pub use network::{Direct, Fifo, InputMode, Layer, Network, OutputRegion, Stream, WeightRegion};
pub use regs::{LayerReg, LayerRegs, Quadrant, Reg};

use crate::gcr::clocks::{Clock, Disabled, Enabled, InternalPll, PeripheralClock};
use crate::gcr::{ClockForPeripheral, GcrRegisters};
use core::marker::PhantomData;
use embedded_hal::delay::DelayNs;

/// Highest clock frequency the accelerator supports, with the datapath pipeline enabled
pub const MAX_PIPELINED_FREQUENCY: u32 = 200_000_000;

/// Highest clock frequency the accelerator supports with the pipeline disabled
pub const MAX_NON_PIPELINED_FREQUENCY: u32 = 50_000_000;

/// Settling time for the CNN power-domain load switches, in milliseconds
pub const LOAD_SWITCH_SETTLE_MS: u32 = 10;

/// Whether the accelerator datapath pipeline is enabled
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Pipeline {
    #[default]
    Enabled,
    Disabled,
}

impl Pipeline {
    pub const fn ctl_bits(self) -> u32 {
        match self {
            Self::Enabled => 0,
            Self::Disabled => 1 << 5,
        }
    }
}

/// Clock source for the accelerator
#[derive(Clone, Copy)]
pub enum CnnClockSource {
    Peripheral(Clock<PeripheralClock>),
    Iso,
    Ipll(Clock<InternalPll>),
}

impl CnnClockSource {
    /// Frequency of this source before the CNN divider is applied.
    pub const fn frequency(&self) -> u32 {
        match self {
            Self::Peripheral(clock) => clock.frequency,
            Self::Iso => 60_000_000,
            Self::Ipll(_) => InternalPll::CNN_FREQUENCY,
        }
    }
}

/// Divider applied to the selected clock source.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CnnClockDiv {
    Div1,
    /// The reset value.
    #[default]
    Div2,
    Div4,
    Div8,
    Div16,
}

impl CnnClockDiv {
    pub const fn divisor(self) -> u32 {
        match self {
            Self::Div1 => 1,
            Self::Div2 => 2,
            Self::Div4 => 4,
            Self::Div8 => 8,
            Self::Div16 => 16,
        }
    }
}

#[doc(hidden)]
pub trait CnnState: crate::Sealed {}
impl CnnState for Disabled {}
impl CnnState for Enabled {}

macro_rules! all_quadrants {
    ($w:expr, $f0:ident, $f1:ident, $f2:ident, $f3:ident, $bit:expr) => {
        $w.$f0()
            .bit($bit)
            .$f1()
            .bit($bit)
            .$f2()
            .bit($bit)
            .$f3()
            .bit($bit)
    };
}

/// CNN accelerator
pub struct Cnn<S: CnnState> {
    cnn: crate::pac::Cnn,
    q0: crate::pac::Cnnx16_0,
    q1: crate::pac::Cnnx16_1,
    q2: crate::pac::Cnnx16_2,
    q3: crate::pac::Cnnx16_3,
    gcfr: crate::pac::Gcfr,
    pipeline: Pipeline,
    source: Option<CnnClockSource>,
    divider: CnnClockDiv,
    _state: PhantomData<S>,
}

impl Cnn<Disabled> {
    /// Take ownership of the accelerator. Does not power it up
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cnn: crate::pac::Cnn,
        q0: crate::pac::Cnnx16_0,
        q1: crate::pac::Cnnx16_1,
        q2: crate::pac::Cnnx16_2,
        q3: crate::pac::Cnnx16_3,
        gcfr: crate::pac::Gcfr,
    ) -> Self {
        Self {
            cnn,
            q0,
            q1,
            q2,
            q3,
            gcfr,
            pipeline: Pipeline::Enabled,
            source: None,
            divider: CnnClockDiv::default(),
            _state: PhantomData,
        }
    }

    /// Disable the datapath pipeline, cap clock frequency
    pub fn with_pipeline(mut self, mode: Pipeline) -> Self {
        self.pipeline = mode;
        self
    }

    /// Power up the accelerator and start its clock
    pub fn enable(
        self,
        reg: &mut GcrRegisters,
        source: CnnClockSource,
        divider: CnnClockDiv,
        delay: &mut impl DelayNs,
    ) -> Cnn<Enabled> {
        let frequency = source.frequency() / divider.divisor();
        debug_assert_frequency(self.pipeline, frequency);

        self.gcfr.reg3().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_rst,
                cnnx16_1_rst,
                cnnx16_2_rst,
                cnnx16_3_rst,
                true
            )
        });
        self.gcfr.reg1().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_ram_en,
                cnnx16_1_ram_en,
                cnnx16_2_ram_en,
                cnnx16_3_ram_en,
                true
            )
        });
        self.gcfr.reg0().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_pwr_en,
                cnnx16_1_pwr_en,
                cnnx16_2_pwr_en,
                cnnx16_3_pwr_en,
                true
            )
        });

        delay.delay_ms(LOAD_SWITCH_SETTLE_MS);

        self.gcfr.reg2().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_iso,
                cnnx16_1_iso,
                cnnx16_2_iso,
                cnnx16_3_iso,
                false
            )
        });
        self.gcfr.reg3().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_rst,
                cnnx16_1_rst,
                cnnx16_2_rst,
                cnnx16_3_rst,
                false
            )
        });

        if matches!(source, CnnClockSource::Ipll(_)) {
            while reg.gcr.ipll_ctrl().read().rdy().bit_is_clear() {}
        }
        write_clock(reg, source, divider);

        unsafe {
            self.cnn.enable_clock(&mut reg.gcr);
        }

        Cnn {
            cnn: self.cnn,
            q0: self.q0,
            q1: self.q1,
            q2: self.q2,
            q3: self.q3,
            gcfr: self.gcfr,
            pipeline: self.pipeline,
            source: Some(source),
            divider,
            _state: PhantomData,
        }
    }
}

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
    pub fn init<M: InputMode>(&mut self, network: &Network<M>) {
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

        let stop = M::STOP_SM | self.pipeline.ctl_bits();
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
    /// The accelerator clock frequency, after the divider
    pub const fn frequency(&self) -> u32 {
        match self.source {
            Some(source) => source.frequency() / self.divider.divisor(),
            None => 0,
        }
    }

    /// The selected clock source
    pub const fn source(&self) -> CnnClockSource {
        match self.source {
            Some(source) => source,
            None => CnnClockSource::Iso,
        }
    }

    /// Change the clock divider without repeating the power-up sequence
    pub fn set_divider(&mut self, reg: &mut GcrRegisters, divider: CnnClockDiv) {
        let source = self.source();
        debug_assert_frequency(self.pipeline, source.frequency() / divider.divisor());
        write_clock(reg, source, divider);
        self.divider = divider;
    }

    /// Gate the clock and power the accelerator down
    pub fn disable(self, reg: &mut GcrRegisters) -> Cnn<Disabled> {
        unsafe {
            self.cnn.disable_clock(&mut reg.gcr);
        }

        self.gcfr.reg3().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_rst,
                cnnx16_1_rst,
                cnnx16_2_rst,
                cnnx16_3_rst,
                true
            )
        });
        self.gcfr.reg2().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_iso,
                cnnx16_1_iso,
                cnnx16_2_iso,
                cnnx16_3_iso,
                true
            )
        });
        self.gcfr.reg0().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_pwr_en,
                cnnx16_1_pwr_en,
                cnnx16_2_pwr_en,
                cnnx16_3_pwr_en,
                false
            )
        });
        self.gcfr.reg1().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_ram_en,
                cnnx16_1_ram_en,
                cnnx16_2_ram_en,
                cnnx16_3_ram_en,
                false
            )
        });
        self.gcfr.reg3().modify(|_, w| {
            all_quadrants!(
                w,
                cnnx16_0_rst,
                cnnx16_1_rst,
                cnnx16_2_rst,
                cnnx16_3_rst,
                false
            )
        });

        Cnn {
            cnn: self.cnn,
            q0: self.q0,
            q1: self.q1,
            q2: self.q2,
            q3: self.q3,
            gcfr: self.gcfr,
            pipeline: self.pipeline,
            source: None,
            divider: self.divider,
            _state: PhantomData,
        }
    }
}

impl<S: CnnState> Cnn<S> {
    /// The configured pipeline mode
    pub const fn pipeline(&self) -> Pipeline {
        self.pipeline
    }

    /// Register file for one quadrant
    pub const fn quadrant(&self, index: u8) -> Quadrant {
        Quadrant::new(index)
    }

    /// Retain the accelerator's contents across UPM, STANDBY and BACKUP
    /// TODO: evaluate power draw on hardware and see if we'd rather just reboot
    pub fn set_retention(&mut self, enable: bool) {
        self.gcfr.reg2().modify(|_, w| {
            let w = all_quadrants!(
                w,
                cnnx16_0_data_ret_en,
                cnnx16_1_data_ret_en,
                cnnx16_2_data_ret_en,
                cnnx16_3_data_ret_en,
                enable
            );
            all_quadrants!(
                w,
                cnnx16_0_ram_data_ret_en,
                cnnx16_1_ram_data_ret_en,
                cnnx16_2_ram_data_ret_en,
                cnnx16_3_ram_data_ret_en,
                enable
            )
        });
    }
}

/// Writes the source and the divider in one register write so we don't overclock the accelerator
fn write_clock(reg: &mut GcrRegisters, source: CnnClockSource, divider: CnnClockDiv) {
    reg.gcr.pclkdiv().modify(|_, w| {
        let w = match divider {
            CnnClockDiv::Div1 => w.cnnclkdiv().div1(),
            CnnClockDiv::Div2 => w.cnnclkdiv().div2(),
            CnnClockDiv::Div4 => w.cnnclkdiv().div4(),
            CnnClockDiv::Div8 => w.cnnclkdiv().div8(),
            CnnClockDiv::Div16 => w.cnnclkdiv().div16(),
        };
        match source {
            CnnClockSource::Peripheral(_) => w.cnnclksel().pclk(),
            CnnClockSource::Iso => w.cnnclksel().iso(),
            CnnClockSource::Ipll(_) => w.cnnclksel().ipll(),
        }
    });
}

#[inline]
fn debug_assert_frequency(pipeline: Pipeline, frequency: u32) {
    debug_assert!(
        frequency <= MAX_PIPELINED_FREQUENCY,
        "CNN clock exceeds the maximum supported frequency"
    );
    debug_assert!(
        matches!(pipeline, Pipeline::Enabled) || frequency <= MAX_NON_PIPELINED_FREQUENCY,
        "CNN clock exceeds the maximum supported frequency without the pipeline"
    );
}
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_order_covers_every_register_once() {
        for expected in regs::ALL_LAYER_REGS {
            let count = EMIT_ORDER.iter().filter(|r| **r == expected).count();
            assert_eq!(count, 1, "{expected:?} appears {count} times in EMIT_ORDER");
        }
        assert_eq!(EMIT_ORDER.len(), regs::ALL_LAYER_REGS.len());
    }

    #[test]
    fn enables_are_written_last() {
        assert_eq!(*EMIT_ORDER.last().unwrap(), LayerReg::En);
    }

    #[test]
    fn oned_is_written_after_output_channel_count() {
        let pos = |r: LayerReg| EMIT_ORDER.iter().position(|x| *x == r).unwrap();
        assert!(pos(LayerReg::Oned) > pos(LayerReg::Ochan));
        assert!(pos(LayerReg::Post) > pos(LayerReg::Tptr));
    }

    #[test]
    fn dividers_match_their_names() {
        assert_eq!(CnnClockDiv::Div1.divisor(), 1);
        assert_eq!(CnnClockDiv::Div2.divisor(), 2);
        assert_eq!(CnnClockDiv::Div4.divisor(), 4);
        assert_eq!(CnnClockDiv::Div8.divisor(), 8);
        assert_eq!(CnnClockDiv::Div16.divisor(), 16);
        // The reset value of PCLKDIV.CNNCLKDIV is div-by-2, not div-by-1.
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
        assert!(
            InternalPll::CNN_FREQUENCY / CnnClockDiv::Div2.divisor() > MAX_NON_PIPELINED_FREQUENCY
        );
    }

    #[test]
    fn pipeline_defaults_to_enabled() {
        assert_eq!(Pipeline::default(), Pipeline::Enabled);
    }

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

    /// The generator folds `NO_PIPELINE` into the stop-SM and arm words, so it
    /// cannot be written once and left alone.
    #[test]
    fn pipeline_contributes_to_every_control_word() {
        assert_eq!(Pipeline::Enabled.ctl_bits(), 0);
        assert_eq!(Pipeline::Disabled.ctl_bits(), 1 << 5);

        // The published constants are the pipelined case.
        for word in [
            network::Direct::STOP_SM,
            network::Direct::START_MASTER,
            network::Fifo::STOP_SM,
            network::Fifo::START_MASTER,
        ] {
            assert_eq!(word & (1 << 5), 0, "{word:#010x} already has NO_PIPELINE");
        }

        assert_eq!(
            network::Direct::STOP_SM | Pipeline::Disabled.ctl_bits(),
            0x0010_0028
        );
        assert_eq!(
            network::Fifo::STOP_SM | Pipeline::Disabled.ctl_bits(),
            0x0010_8028
        );
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
