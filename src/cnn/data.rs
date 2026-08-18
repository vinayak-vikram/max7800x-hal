//! Weights, bias, and moving data in and out

use super::memory;
use super::network::Network;
use super::Cnn;
use crate::gcr::clocks::Enabled;

impl Cnn<Enabled> {
    /// Load a network's weights into kernel memory.
    pub fn load_weights(&mut self, network: &Network) {
        for region in network.weights {
            memory::write_kernel(
                region.quadrant,
                region.processor,
                region.offset,
                region.data,
            );
        }
    }

    /// Load a network's bias values, if it has any.
    pub fn load_bias(&mut self, network: &Network) {
        let Some(tables) = network.bias else {
            return;
        };
        for (quadrant, data) in tables.iter().enumerate() {
            memory::write_bias(quadrant as u8, data);
        }
    }

    /// Place a network's 32-bit input in data memory. Must precede `start`.
    pub fn write_u32(&mut self, network: &Network, src: &[u32]) -> usize {
        let needed = network.input_words();
        assert!(
            src.len() >= needed,
            "input buffer holds {} words, network takes {}",
            src.len(),
            needed
        );

        let mut pos = 0;
        for region in network.input {
            let len = region.len as usize;
            memory::write_data(
                region.quadrant,
                region.instance,
                region.word as u32,
                &src[pos..pos + len],
            );
            pos += len;
        }
        pos
    }

    /// Place a network's 8-bit input in data memory, four channels per word.
    ///
    /// Only `(N, 1, 1)` shapes. Anything with spatial extent, an image say, is
    /// one word per pixel and would need interleaving from CHW first.
    /// TODO: handle spatial 8-bit input
    pub fn write_u8(&mut self, network: &Network, src: &[u8]) -> usize {
        let needed = network.input_words() * 4;
        assert!(
            src.len() >= needed,
            "input buffer holds {} bytes, network takes {}",
            src.len(),
            needed
        );

        let mut pos = 0;
        for region in network.input {
            for word in 0..region.len as u32 {
                let packed =
                    u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]]);
                memory::write_data(
                    region.quadrant,
                    region.instance,
                    region.word as u32 + word,
                    &[packed],
                );
                pos += 4;
            }
        }
        pos
    }

    /// Read a network's 32-bit output.
    pub fn read_u32(&self, network: &Network, dst: &mut [u32]) -> usize {
        let needed = network.output_words();
        assert!(
            dst.len() >= needed,
            "output buffer holds {} words, network produces {}",
            dst.len(),
            needed
        );

        let mut pos = 0;
        for region in network.output {
            let len = region.len as usize;
            memory::read_data(
                region.quadrant,
                region.instance,
                region.word as u32,
                &mut dst[pos..pos + len],
            );
            pos += len;
        }
        pos
    }

    /// Read a network's 8-bit output, four channels per word.
    ///
    /// Only `(N, 1, 1)` shapes. Anything with spatial extent comes back
    /// pixel-major, not CHW, and would need de-interleaving.
    /// TODO: handle spatial 8-bit output
    pub fn read_u8(&self, network: &Network, dst: &mut [u8]) -> usize {
        let needed = network.output_words() * 4;
        assert!(
            dst.len() >= needed,
            "output buffer holds {} bytes, network produces {}",
            dst.len(),
            needed
        );

        let mut pos = 0;
        for region in network.output {
            for word in 0..region.len as u32 {
                let mut buf = [0u32; 1];
                memory::read_data(
                    region.quadrant,
                    region.instance,
                    region.word as u32 + word,
                    &mut buf,
                );
                dst[pos..pos + 4].copy_from_slice(&buf[0].to_le_bytes());
                pos += 4;
            }
        }
        pos
    }
}
