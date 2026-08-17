//! Full network descriptor

use super::fields::{
    Ccnt, Ena, Lctl, Lctl2, Mcnt1, Mcnt2, Nxtlyr, Ochan, Oned, Pccnt, Post, Prcnt, Rcnt, RptrBase,
    Stride, Tptr, WptrBase, WptrChoffs, WptrMoffs, WptrToffs,
};
use super::regs::QUADRANTS;

// The accelerator can also take its input through a FIFO, which is what
// streaming layers require. Neither is implemented; the target workload feeds
// buffered sample windows that are already in RAM. This is kept, commented,
// because the `CTL` words below are verified against generated `cnn.c` and are
// the hard part of reviving it: the direct values against `kws20_demo`, the
// FIFO values against `mobilefacenet-112`. An earlier revision of the register
// spec had bit 0 inverted, which hangs the accelerator.
//
// They assume the configuration every shipped network uses: pipeline enabled,
// memory-express weight loading, ready-select 0, quadrant 0 as master, and no
// snoop, one-shot or fast FIFO.
//
// pub trait InputMode: crate::Sealed {
//     const FIFO: bool;
//     const STOP_SM: u32;
//     const START_MASTER: u32;
//     const START_OTHER: u32;
//     const START_GO: u32;
// }
//
// pub struct Direct;
// pub struct Fifo;
//
// impl InputMode for Direct {
//     const FIFO: bool = false;
//     const STOP_SM: u32 = 0x0010_0008;
//     const START_MASTER: u32 = 0x0010_0808;
//     const START_OTHER: u32 = 0x0010_0809;
//     const START_GO: u32 = 0x0010_0009;
// }
//
// impl InputMode for Fifo {
//     const FIFO: bool = true;
//     const STOP_SM: u32 = 0x0010_8008;
//     const START_MASTER: u32 = 0x0018_c808;
//     const START_OTHER: u32 = 0x0018_c809;
//     const START_GO: u32 = 0x0018_c809;
// }
//
// FIFO mode sets FIFO_EN (bit 15) in every word and STREAM_EN (14) plus
// STREAM_FIFO (19) once armed; its go word equals its non-master arm word
// because it keeps EXT_SYNC (11), which direct mode drops.

/// One hardware layer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
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

    /// program into master quadrant only?
    pub master_only: bool,
}

impl Layer {
    /// A layer that configures nothing, for `..Layer::new()` in a `const`
    pub const fn new() -> Self {
        Self {
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
            lctl: [Lctl::new(); QUADRANTS as usize],
            post: [Post::new(); QUADRANTS as usize],
            wptr: [WptrBase::new(); QUADRANTS as usize],
            ena: [Ena::new(); QUADRANTS as usize],
            master_only: false,
        }
    }

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
pub struct Network<'a> {
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
}

impl<'a> Network<'a> {
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
        }
    }

    /// Whether any layer streams. Streaming requires FIFO input.
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
        let net: Network = Network::new(&[], 0, 0, &[], None, &[], &OUTPUT);
        assert_eq!(net.output_words(), 21);

        // mobilefacenet-112: 64 channels of 8-bit output as 16 single words,
        // one per instance across all four quadrants.
        const BYTES: [OutputRegion; 16] = [OutputRegion {
            quadrant: 0,
            instance: 0,
            word: 10240,
            len: 1,
        }; 16];
        let net: Network = Network::new(&[], 0, 0, &[], None, &[], &BYTES);
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
        let net: Network = Network::new(&[], 0, 0, &[], None, &regions, &[]);
        assert_eq!(net.input_words(), 12 * 7200);

        let net: Network = Network::new(&[], 0, 0, &[], None, &[], &[]);
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

    /// `new` is the const-context form of `Default`, and generated networks
    /// use it with struct update syntax, so the two must not drift.
    #[test]
    fn new_matches_default() {
        assert_eq!(Layer::new(), Layer::default());
    }
}
