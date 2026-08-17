//! Checks a network before anything reaches the accelerator

use super::config::MASTER_QUADRANT;
use super::memory::{bias_fits, kernel_burst_fits, DATA_WINDOW_BYTES};
use super::network::{InputMode, Network};
use super::regs::{DATA_INSTANCES_PER_QUADRANT, MAX_LAYERS, PROCESSORS_PER_QUADRANT, QUADRANTS};

/// Why a network cannot be programmed
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Invalid {
    /// More layers than the register file holds
    TooManyLayers { layers: usize },
    /// `first_layer`/`last_layer` do not index the layer table
    LayerRange { first: u8, last: u8, layers: usize },
    /// A weight region names a quadrant or processor that does not exist
    WeightTarget { region: usize },
    /// A weight region runs past the end of a processor's kernel window
    WeightOverrun { region: usize },
    /// More bias values than a quadrant's bias memory holds
    BiasOverrun { quadrant: u8 },
    /// A data region names a quadrant or instance that does not exist
    DataTarget { region: usize, input: bool },
    /// A data region runs past the end of its memory window
    DataOverrun { region: usize, input: bool },
    /// `MASK_ENA` is neither zero nor a mirror of `PROC_ENA`
    MaskEnables { layer: u8, quadrant: u8 },
    /// `SIENA` is set outside the master quadrant, or names the master itself
    SourceEnables { layer: u8, quadrant: u8 },
    /// `SHIFT_CNT` has spilled into the bit `DW_BCAST` occupies
    ShiftCountCollision { layer: u8, quadrant: u8 },
}

impl<M: InputMode> Network<'_, M> {
    /// Check the network against everything knowable without hardware
    pub fn validate(&self) -> Result<(), Invalid> {
        self.check_layer_table()?;
        self.check_weights()?;
        self.check_bias()?;
        self.check_data()?;
        self.check_layers()
    }

    fn check_layer_table(&self) -> Result<(), Invalid> {
        let layers = self.layers.len();
        if layers > MAX_LAYERS as usize {
            return Err(Invalid::TooManyLayers { layers });
        }
        if self.first_layer > self.last_layer || self.last_layer as usize >= layers.max(1) {
            return Err(Invalid::LayerRange {
                first: self.first_layer,
                last: self.last_layer,
                layers,
            });
        }
        Ok(())
    }

    fn check_weights(&self) -> Result<(), Invalid> {
        for (region, w) in self.weights.iter().enumerate() {
            if w.quadrant >= QUADRANTS || w.processor >= PROCESSORS_PER_QUADRANT {
                return Err(Invalid::WeightTarget { region });
            }
            if !w.offset.is_multiple_of(4) || !kernel_burst_fits(w.offset, w.data.len()) {
                return Err(Invalid::WeightOverrun { region });
            }
        }
        Ok(())
    }

    fn check_bias(&self) -> Result<(), Invalid> {
        let Some(tables) = self.bias else {
            return Ok(());
        };
        for (quadrant, data) in tables.iter().enumerate() {
            if !bias_fits(data.len()) {
                return Err(Invalid::BiasOverrun {
                    quadrant: quadrant as u8,
                });
            }
        }
        Ok(())
    }

    fn check_data(&self) -> Result<(), Invalid> {
        let check = |quadrant: u8, instance: u8, word: u16, len: u16, region, input| {
            if quadrant >= QUADRANTS || instance >= DATA_INSTANCES_PER_QUADRANT {
                return Err(Invalid::DataTarget { region, input });
            }
            let end = (word as u64 + len as u64) * 4;
            if end > DATA_WINDOW_BYTES as u64 {
                return Err(Invalid::DataOverrun { region, input });
            }
            Ok(())
        };
        for (region, r) in self.input.iter().enumerate() {
            check(r.quadrant, r.instance, r.word, r.len, region, true)?;
        }
        for (region, r) in self.output.iter().enumerate() {
            check(r.quadrant, r.instance, r.word, r.len, region, false)?;
        }
        Ok(())
    }

    fn check_layers(&self) -> Result<(), Invalid> {
        for (index, layer) in self.layers.iter().enumerate() {
            let index = index as u8;
            for q in 0..QUADRANTS {
                let (lctl, post, ena) = (
                    layer.lctl[q as usize],
                    layer.post[q as usize],
                    layer.ena[q as usize],
                );

                let mask = ena.mask_ena();
                if mask != 0 && mask != ena.proc_ena() {
                    return Err(Invalid::MaskEnables {
                        layer: index,
                        quadrant: q,
                    });
                }

                let siena = lctl.siena();
                let names_master = siena & (1 << MASTER_QUADRANT) != 0;
                if (siena != 0 && q != MASTER_QUADRANT) || names_master {
                    return Err(Invalid::SourceEnables {
                        layer: index,
                        quadrant: q,
                    });
                }

                // Spec 8.4: with tcalc clear the field holds in_expand - 1, and
                // a value >= 8 sets bit 29 on its own, which the hardware reads
                // as DW_BCAST.
                if lctl.rd_ahead() && !post.tcalc() && lctl.shift_cnt() >= 8 {
                    return Err(Invalid::ShiftCountCollision {
                        layer: index,
                        quadrant: q,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::fields::*;
    use crate::cnn::network::{Direct, InputRegion, Layer, OutputRegion, WeightRegion};

    fn net(layers: &[Layer]) -> Network<'_, Direct> {
        Network::new(
            layers,
            0,
            (layers.len() as u8).saturating_sub(1),
            &[],
            None,
            &[],
            &[],
        )
    }

    #[test]
    fn a_blank_layer_is_valid() {
        assert_eq!(net(&[Layer::default()]).validate(), Ok(()));
    }

    /// `kws20_demo` layer 0: `LCTL = 0xeb20` on the master naming quadrants
    /// 1-3, `0x0b20` elsewhere, and `ENA = 0xffffffff`.
    #[test]
    fn the_shipped_shape_validates() {
        let mut layer = Layer::default();
        layer.lctl = [
            Lctl::from_bits(0x0000_eb20),
            Lctl::from_bits(0x0000_0b20),
            Lctl::from_bits(0x0000_0b20),
            Lctl::from_bits(0x0000_0b20),
        ];
        layer.ena = [Ena::from_bits(0xffff_ffff); 4];
        assert_eq!(net(&[layer]).validate(), Ok(()));
    }

    /// Every `ENA` value in the eight shipped examples is one of these, and
    /// all of them satisfy the rule.
    #[test]
    fn shipped_enable_words_pass() {
        for bits in [
            0x0000_000fu32,
            0x0000_00ff,
            0x0000_0fff,
            0x0000_ffff,
            0x0007_0007,
            0x000f_000f,
            0x00ff_00ff,
            0x0fff_0fff,
            0x7000_7000,
            0xf000_f000,
            0xffff_ffff,
        ] {
            let mut layer = Layer::default();
            layer.ena = [Ena::from_bits(bits); 4];
            assert_eq!(net(&[layer]).validate(), Ok(()), "{bits:#010x}");
        }
    }

    /// A mask enable that is neither zero nor a mirror means the weight
    /// memories armed do not match the processors doing the work.
    #[test]
    fn mismatched_mask_enables_are_rejected() {
        let mut layer = Layer::default();
        layer.ena = [Ena::from_bits(0x000f_00ff); 4];
        assert_eq!(
            net(&[layer]).validate(),
            Err(Invalid::MaskEnables {
                layer: 0,
                quadrant: 0
            })
        );
    }

    /// `SIENA` lives only in the master's word, and never names the master.
    #[test]
    fn source_enables_belong_to_the_master() {
        let mut layer = Layer::default();
        layer.lctl[1] = Lctl::new().with_siena(0b1110);
        assert_eq!(
            net(&[layer]).validate(),
            Err(Invalid::SourceEnables {
                layer: 0,
                quadrant: 1
            })
        );

        let mut layer = Layer::default();
        layer.lctl[0] = Lctl::new().with_siena(0b1111);
        assert_eq!(
            net(&[layer]).validate(),
            Err(Invalid::SourceEnables {
                layer: 0,
                quadrant: 0
            })
        );
    }

    /// Spec 8.4. Only the three conditions together are a fault: 688 shipped
    /// layers set `rd_ahead` with `tcalc` clear, and 180 carry a shift count
    /// of 8 or more, but no shipped layer does both.
    #[test]
    fn read_ahead_shift_collision_is_rejected() {
        let bad = Lctl::new().with_rd_ahead(true).with_shift_cnt(8);
        let mut layer = Layer::default();
        layer.lctl = [bad; 4];
        assert_eq!(
            net(&[layer]).validate(),
            Err(Invalid::ShiftCountCollision {
                layer: 0,
                quadrant: 0
            })
        );

        // Seven fits in the three bits below DW_BCAST.
        let mut layer = Layer::default();
        layer.lctl = [Lctl::new().with_rd_ahead(true).with_shift_cnt(7); 4];
        assert_eq!(net(&[layer]).validate(), Ok(()));

        // With tcalc the field is a different, smaller quantity.
        let mut layer = Layer::default();
        layer.lctl = [bad; 4];
        layer.post = [Post::new().with_tcalc(true); 4];
        assert_eq!(net(&[layer]).validate(), Ok(()));

        // Read-ahead is what makes the field mean in_expand at all.
        let mut layer = Layer::default();
        layer.lctl = [Lctl::new().with_shift_cnt(15); 4];
        assert_eq!(net(&[layer]).validate(), Ok(()));
    }

    #[test]
    fn layer_bounds_are_checked() {
        let layers = [Layer::default(), Layer::default()];
        let n: Network<Direct> = Network::new(&layers, 1, 0, &[], None, &[], &[]);
        assert_eq!(
            n.validate(),
            Err(Invalid::LayerRange {
                first: 1,
                last: 0,
                layers: 2
            })
        );

        let n: Network<Direct> = Network::new(&layers, 0, 2, &[], None, &[], &[]);
        assert!(matches!(n.validate(), Err(Invalid::LayerRange { .. })));
    }

    #[test]
    fn weight_regions_are_bounds_checked() {
        const DATA: [u32; 4] = [0; 4];

        let bad = [WeightRegion {
            quadrant: 4,
            processor: 0,
            offset: 0,
            data: &DATA,
        }];
        let n: Network<Direct> = Network::new(&[], 0, 0, &bad, None, &[], &[]);
        assert_eq!(n.validate(), Err(Invalid::WeightTarget { region: 0 }));

        // One word short of the window is fine; the window itself is not.
        let bad = [WeightRegion {
            quadrant: 0,
            processor: 0,
            offset: crate::cnn::memory::KERNEL_WINDOW_BYTES - 4,
            data: &DATA,
        }];
        let n: Network<Direct> = Network::new(&[], 0, 0, &bad, None, &[], &[]);
        assert_eq!(n.validate(), Err(Invalid::WeightOverrun { region: 0 }));
    }

    #[test]
    fn data_regions_are_bounds_checked() {
        let bad = [InputRegion {
            quadrant: 0,
            instance: 4,
            word: 0,
            len: 1,
        }];
        let n: Network<Direct> = Network::new(&[], 0, 0, &[], None, &bad, &[]);
        assert_eq!(
            n.validate(),
            Err(Invalid::DataTarget {
                region: 0,
                input: true
            })
        );

        let bad = [OutputRegion {
            quadrant: 0,
            instance: 0,
            word: 32767,
            len: 4,
        }];
        let n: Network<Direct> = Network::new(&[], 0, 0, &[], None, &[], &bad);
        assert_eq!(
            n.validate(),
            Err(Invalid::DataOverrun {
                region: 0,
                input: false
            })
        );
    }

    #[test]
    fn bias_tables_are_bounds_checked() {
        const BIG: [u8; 2049] = [0; 2049];
        const TABLES: [&[u8]; 4] = [&[], &BIG, &[], &[]];
        let n: Network<Direct> = Network::new(&[], 0, 0, &[], Some(&TABLES), &[], &[]);
        assert_eq!(n.validate(), Err(Invalid::BiasOverrun { quadrant: 1 }));
    }
}
