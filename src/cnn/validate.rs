//! Checks a network before anything reaches the accelerator

use super::config::{emit_layer, LayerSink, MASTER_QUADRANT};
use super::fields::reserved_bits;
use super::memory::{bias_fits, kernel_burst_fits, DATA_WINDOW_BYTES};
use super::network::{Layer, Network};
use super::regs::LayerReg;
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
    /// A register has bits set that belong to no field
    ReservedBits {
        layer: u8,
        quadrant: u8,
        reg: LayerReg,
        bits: u32,
    },
}

impl Network<'_> {
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

                check_reserved(layer, index, q)?;
            }
        }
        Ok(())
    }
}

/// Rejects any register in one layer-quadrant holding a bit no field covers
fn check_reserved(layer: &Layer, index: u8, quadrant: u8) -> Result<(), Invalid> {
    let mut found = Ok(());
    emit_layer(
        &mut ReservedCheck {
            index,
            quadrant,
            found: &mut found,
        },
        layer,
        quadrant,
    );
    found
}

struct ReservedCheck<'a> {
    index: u8,
    quadrant: u8,
    found: &'a mut Result<(), Invalid>,
}

impl LayerSink for ReservedCheck<'_> {
    fn write(&mut self, reg: LayerReg, bits: u32) {
        let reserved = reserved_bits(reg, bits);
        if reserved != 0 && self.found.is_ok() {
            *self.found = Err(Invalid::ReservedBits {
                layer: self.index,
                quadrant: self.quadrant,
                reg,
                bits: reserved,
            });
        }
    }
}
