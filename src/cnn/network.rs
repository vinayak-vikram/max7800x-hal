//! Full network descriptor

use super::fields::{
    Ccnt, Ena, Fmax, Lctl, Lctl2, Mcnt1, Mcnt2, Nxtlyr, Ochan, Oned, Pccnt, Post, Prcnt, Rcnt,
    RptrBase, Stream1, Stream2, Stride, Tptr, WptrBase, WptrChoffs, WptrMoffs, WptrToffs,
};
use super::regs::QUADRANTS;
use core::marker::PhantomData;

/// How input data reaches the accelerator.
pub trait InputMode: crate::Sealed {
    /// Whether input arrives through the FIFO.
    const FIFO: bool;
    /// `CTL` value that halts the state machine, written during init.
    const STOP_SM: u32;
    /// `CTL` value that arms the master quadrant. Note `CNN_EN` is **clear**
    /// here; setting it would start the master before the others are armed.
    const START_MASTER: u32;
    /// `CTL` value that arms a non-master quadrant, with `CNN_EN` set.
    const START_OTHER: u32;
    /// `CTL` value written to the master last, which starts the network.
    const START_GO: u32;
}

pub struct Direct;
pub struct Fifo;

impl crate::Sealed for Direct {}
impl crate::Sealed for Fifo {}

impl InputMode for Direct {
    const FIFO: bool = false;
    const STOP_SM: u32 = 0x0010_0008;
    const START_MASTER: u32 = 0x0010_0808;
    const START_OTHER: u32 = 0x0010_0809;
    const START_GO: u32 = 0x0010_0009;
}

impl InputMode for Fifo {
    const FIFO: bool = true;
    const STOP_SM: u32 = 0x0010_8008;
    const START_MASTER: u32 = 0x0018_c808;
    const START_OTHER: u32 = 0x0018_c809;
    const START_GO: u32 = 0x0018_c809;
}

/// Streaming configuration for one layer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stream {
    pub slot: u8,
    pub start: Stream1,
    pub delta: Stream2,
    pub rollover: Fmax,
}

/// One hardware layer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Layer {
    pub next: Nxtlyr,
    pub rows: Rcnt,
    pub cols: Ccnt,
    pub oned: Oned,
    pub pool_rows: Prcnt,
    pub pool_cols: Pccnt,
    pub stride: Stride,
    pub wptr_ts: WptrToffs,
    pub wptr_moffs: Option<WptrMoffs>,
    pub wptr_choffs: WptrChoffs,
    pub rptr: RptrBase,
    pub lctl2: Lctl2,
    pub mcnt1: Option<Mcnt1>,
    pub mcnt2: Mcnt2,
    pub ochan: Ochan,
    pub tptr: Tptr,

    pub lctl: [Lctl; QUADRANTS as usize],
    pub post: [Post; QUADRANTS as usize],
    pub wptr: [WptrBase; QUADRANTS as usize],
    pub ena: [Ena; QUADRANTS as usize],

    pub stream: Option<Stream>,
    /// program into master quadrant only?
    pub master_only: bool,
}

impl Layer {
    // just for debug
    pub fn is_synthetic(&self) -> bool {
        self.rows.bits() == 0x0001_0000
            && self.cols.bits() == 0x0001_0000
            && self.stride.bits() == 0x0000_0010
            && self.oned.bits() == 0x0000_0100
            && self.lctl[0].bits() == 0x0000_0920
            && self.post[0].bits() == 0x0300_0000
            && self.ena[0].bits() == 0x0000_0001
    }
}

/// A contiguous run of weights destined for one processor's kernel memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WeightRegion<'a> {
    pub quadrant: u8,
    pub processor: u8,
    /// Byte offset within the processor's kernel memory window.
    pub offset: u32,
    pub data: &'a [u32],
}

/// A region of data memory the network reads its input from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InputRegion {
    pub quadrant: u8,
    pub instance: u8,
    /// Word offset within the memory instance.
    pub word: u16,
    /// Length in words.
    pub len: u16,
}

/// A region of data memory holding part of the network's output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OutputRegion {
    pub quadrant: u8,
    pub instance: u8,
    /// Word offset within the memory instance.
    pub word: u16,
    /// Length in words.
    pub len: u16,
}

/// Hello gee.
pub struct Network<'a, M: InputMode> {
    pub layers: &'a [Layer],
    /// Index of the first hardware layer to execute
    pub first_layer: u8,
    /// Index of the last hardware layer
    pub last_layer: u8,
    pub weights: &'a [WeightRegion<'a>],
    /// Bias data per quadrant
    pub bias: Option<&'a [&'a [u8]; QUADRANTS as usize]>,
    /// Where the input goes in data memory, before the network starts
    pub input: &'a [InputRegion],
    pub output: &'a [OutputRegion],
    _mode: PhantomData<M>,
}

impl<'a, M: InputMode> Network<'a, M> {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        layers: &'a [Layer],
        first_layer: u8,
        last_layer: u8,
        weights: &'a [WeightRegion<'a>],
        bias: Option<&'a [&'a [u8]; QUADRANTS as usize]>,
        input: &'a [InputRegion],
        output: &'a [OutputRegion],
    ) -> Self {
        Self {
            layers,
            first_layer,
            last_layer,
            weights,
            bias,
            input,
            output,
            _mode: PhantomData,
        }
    }

    /// Whether any layer streams. Streaming requires FIFO input.
    pub fn is_streaming(&self) -> bool {
        self.layers.iter().any(|l| l.stream.is_some())
    }

    pub const fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Total words the input regions cover.
    pub fn input_words(&self) -> usize {
        self.input.iter().map(|r| r.len as usize).sum()
    }

    /// Total words the output regions cover.
    pub fn output_words(&self) -> usize {
        self.output.iter().map(|r| r.len as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verified against generated `cnn.c`: `kws20_demo` for direct mode,
    /// `mobilefacenet-112` for FIFO. The spec records an earlier revision that
    /// had bit 0 inverted, which would have hung the accelerator.
    #[test]
    fn control_words_match_the_generated_sources() {
        assert_eq!(Direct::STOP_SM, 0x0010_0008);
        assert_eq!(Direct::START_MASTER, 0x0010_0808);
        assert_eq!(Direct::START_OTHER, 0x0010_0809);
        assert_eq!(Direct::START_GO, 0x0010_0009);

        assert_eq!(Fifo::STOP_SM, 0x0010_8008);
        assert_eq!(Fifo::START_MASTER, 0x0018_c808);
        assert_eq!(Fifo::START_OTHER, 0x0018_c809);
        assert_eq!(Fifo::START_GO, 0x0018_c809);
    }

    /// The polarity that matters: the master is armed with `CNN_EN` clear and
    /// every other quadrant with it set. Inverting this hangs the network.
    #[test]
    fn master_arms_with_enable_clear() {
        for (master, other) in [
            (Direct::START_MASTER, Direct::START_OTHER),
            (Fifo::START_MASTER, Fifo::START_OTHER),
        ] {
            assert_eq!(master & 1, 0, "master must arm with CNN_EN clear");
            assert_eq!(other & 1, 1, "other quadrants must arm with CNN_EN set");
            assert_eq!(master | 1, other, "the two differ only in CNN_EN");
        }
    }

    /// Both modes keep the APB clock alive so registers stay readable while
    /// the state machine runs, and both use memory-express weight loading.
    #[test]
    fn every_control_word_keeps_clocks_on() {
        for word in [
            Direct::STOP_SM,
            Direct::START_MASTER,
            Direct::START_OTHER,
            Direct::START_GO,
            Fifo::STOP_SM,
            Fifo::START_MASTER,
            Fifo::START_OTHER,
            Fifo::START_GO,
        ] {
            assert_ne!(word & (1 << 3), 0, "{word:#010x} has CLK_EN clear");
            assert_ne!(word & (1 << 20), 0, "{word:#010x} has MEXPRESS clear");
        }
    }

    /// FIFO mode sets the FIFO enable everywhere and streaming only once
    /// armed; direct mode sets neither.
    #[test]
    fn fifo_mode_sets_the_fifo_bits() {
        assert_ne!(Fifo::STOP_SM & (1 << 15), 0, "FIFO_EN");
        assert_eq!(Direct::STOP_SM & (1 << 15), 0);

        assert_ne!(Fifo::START_MASTER & (1 << 14), 0, "STREAM_EN");
        assert_ne!(Fifo::START_MASTER & (1 << 19), 0, "STREAM_FIFO");
        assert_eq!(Direct::START_MASTER & ((1 << 14) | (1 << 19)), 0);
    }

    /// Direct mode drops `EXT_SYNC` on the go word; FIFO mode keeps it, which
    /// is why its go word equals its non-master arm word.
    #[test]
    fn go_word_differs_between_modes() {
        assert_eq!(Direct::START_GO & (1 << 11), 0);
        assert_ne!(Fifo::START_GO & (1 << 11), 0);
        assert_eq!(Fifo::START_GO, Fifo::START_OTHER);
        assert_ne!(Direct::START_GO, Direct::START_OTHER);
    }

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
            stream: None,
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

    #[test]
    fn network_reports_streaming_and_bias() {
        const LAYERS: [Layer; 0] = [];
        let net: Network<Direct> = Network::new(&LAYERS, 0, 0, &[], None, &[], &[]);
        assert!(!net.is_streaming());
        assert!(!net.has_bias());

        let layers = [synthetic_layer()];
        let net: Network<Direct> = Network::new(&layers, 0, 0, &[], None, &[], &[]);
        assert!(!net.is_streaming());

        let mut streaming = synthetic_layer();
        streaming.stream = Some(Stream {
            slot: 0,
            start: Stream1::from_bits(0x147),
            delta: Stream2::from_bits(0x0142_0022),
            rollover: Fmax::from_bits(0x148),
        });
        let layers = [streaming];
        let net: Network<Fifo> = Network::new(&layers, 0, 0, &[], None, &[], &[]);
        assert!(net.is_streaming());
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
        let net: Network<Direct> = Network::new(&[], 0, 0, &[], None, &[], &OUTPUT);
        assert_eq!(net.output_words(), 21);

        // mobilefacenet-112: 64 channels of 8-bit output as 16 single words,
        // one per instance across all four quadrants.
        const BYTES: [OutputRegion; 16] = [OutputRegion {
            quadrant: 0,
            instance: 0,
            word: 10240,
            len: 1,
        }; 16];
        let net: Network<Fifo> = Network::new(&[], 0, 0, &[], None, &[], &BYTES);
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
        let net: Network<Direct> = Network::new(&[], 0, 0, &[], None, &regions, &[]);
        assert_eq!(net.input_words(), 12 * 7200);

        let net: Network<Direct> = Network::new(&[], 0, 0, &[], None, &[], &[]);
        assert_eq!(net.input_words(), 0);
    }

    /// The descriptors must resolve to the addresses the generated
    /// `load_input` pokes: `cifar-100-mobilenet-v2-0.75` loads 1024 words at
    /// `0x51800000`, `imagenet` 12544 at `0x54860000`, and `kinetics` twelve
    /// runs of 7200 at word 960 of each instance.
    #[test]
    fn input_regions_match_the_generated_addresses() {
        use crate::cnn::memory::data_addr;

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
}
