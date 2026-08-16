//! CNN Accelerator Memories
//!

use super::regs::{quadrant_base, DATA_INSTANCES_PER_QUADRANT, PROCESSORS_PER_QUADRANT, QUADRANTS};

/// Offset of the bias memory within a quadrant.
const BIAS_BASE: u32 = 0x0018_0000;
/// Offset of the TRAM within a quadrant.
const TRAM_BASE: u32 = 0x0020_0000;
/// Offset of the kernel memory within a quadrant.
const KERNEL_BASE: u32 = 0x0040_0000;
/// Offset of the data SRAM within a quadrant.
const DATA_BASE: u32 = 0x0080_0000;
/// Address stride between kernel memories, and between data SRAM instances.
const MEMORY_STRIDE: u32 = 0x0002_0000;
/// Address stride between TRAM instances.
const TRAM_STRIDE: u32 = 0x0001_0000;
/// Bias entries per quadrant.
pub const BIAS_ENTRIES: u32 = 2048;
/// TRAM words per processor.
pub const TRAM_WORDS: u32 = 12288;
/// Address space of one data SRAM window, shared by four processors.
pub const DATA_WINDOW_BYTES: u32 = MEMORY_STRIDE;
/// Address space allotted to one processor within a data SRAM window, in words.
pub const DATA_INSTANCE_STRIDE_WORDS: u32 = 8192;
/// Words actually backed by memory within one processor's slot.
pub const DATA_INSTANCE_WORDS: u32 = 5120;
/// Total kernel memory across the whole accelerator, in bytes.
pub const TOTAL_KERNEL_BYTES: u32 = 2_396_160;
/// Kernels available to a processor that is not first in its quadrant.
pub const KERNELS_PER_PROCESSOR: u32 = 4096;
/// Kernels available to the first processor of each quadrant.
/// Extra is for input-layer processing.
pub const KERNELS_FIRST_PROCESSOR: u32 = KERNELS_PER_PROCESSOR + 1024;

/// Address of the bias memory for `quadrant`.
#[inline]
pub const fn bias_addr(quadrant: u8) -> u32 {
    debug_assert!(quadrant < QUADRANTS);
    quadrant_base(quadrant) + BIAS_BASE
}

/// Address of the TRAM for `processor` within `quadrant`.
#[inline]
pub const fn tram_addr(quadrant: u8, processor: u8) -> u32 {
    debug_assert!(quadrant < QUADRANTS);
    debug_assert!(processor < PROCESSORS_PER_QUADRANT);
    quadrant_base(quadrant) + TRAM_BASE + (processor as u32) * TRAM_STRIDE
}

/// Address of the kernel memory for `processor` within `quadrant`.
#[inline]
pub const fn kernel_addr(quadrant: u8, processor: u8) -> u32 {
    debug_assert!(quadrant < QUADRANTS);
    debug_assert!(processor < PROCESSORS_PER_QUADRANT);
    quadrant_base(quadrant) + KERNEL_BASE + (processor as u32) * MEMORY_STRIDE
}

/// Address of data SRAM `instance` within `quadrant`.
/// An instance is the memory shared by four processors.
#[inline]
pub const fn data_addr(quadrant: u8, instance: u8) -> u32 {
    debug_assert!(quadrant < QUADRANTS);
    debug_assert!(instance < DATA_INSTANCES_PER_QUADRANT);
    quadrant_base(quadrant) + DATA_BASE + (instance as u32) * MEMORY_STRIDE
}

/// Number of kernels addressable by `processor` within its quadrant
#[inline]
pub const fn kernel_capacity(processor: u8) -> u32 {
    if processor == 0 {
        KERNELS_FIRST_PROCESSOR
    } else {
        KERNELS_PER_PROCESSOR
    }
}

/// Arm the auto-incrementing kernel write pointer
#[inline]
pub unsafe fn arm_kernel_ptr(addr: u32) {
    ((addr | 1) as *mut u8).write_volatile(0x01);
}

/// Copy words into a data SRAM instance
/// Panics if the access would leave the instance's address window.
pub fn write_data(quadrant: u8, instance: u8, word_offset: u32, src: &[u32]) {
    let end = (word_offset as u64 + src.len() as u64) * 4;
    assert!(
        end <= DATA_WINDOW_BYTES as u64,
        "data SRAM write of {} words at offset {} leaves the {} byte window",
        src.len(),
        word_offset,
        DATA_WINDOW_BYTES
    );
    let base = (data_addr(quadrant, instance) + word_offset * 4) as *mut u32;
    for (i, word) in src.iter().enumerate() {
        unsafe { base.add(i).write_volatile(*word) };
    }
}

/// Address space of one processor's kernel memory window.
pub const KERNEL_WINDOW_BYTES: u32 = MEMORY_STRIDE;

#[inline]
pub const fn kernel_burst_fits(offset: u32, words: usize) -> bool {
    (offset as u64) + (words as u64) * 4 <= KERNEL_WINDOW_BYTES as u64
}

/// Push a burst of packed kernel words into a processor's kernel memory.
pub fn write_kernel(quadrant: u8, processor: u8, offset: u32, data: &[u32]) {
    assert!(
        offset.is_multiple_of(4),
        "kernel burst offset {offset:#x} is not word-aligned"
    );
    assert!(
        kernel_burst_fits(offset, data.len()),
        "kernel burst of {} words at offset {:#x} leaves the {} byte window",
        data.len(),
        offset,
        KERNEL_WINDOW_BYTES
    );

    let base = kernel_addr(quadrant, processor) + offset;
    // SAFETY: the address is inside a kernel window of a quadrant the caller
    // owns, and the burst has been bounds-checked above.
    unsafe { arm_kernel_ptr(base) };
    for (i, word) in data.iter().enumerate() {
        unsafe { ((base + (i as u32) * 4) as *mut u32).write_volatile(*word) };
    }
}

/// Copy words out of a data SRAM instance
/// Panics if the access would leave the instance's address window
pub fn read_data(quadrant: u8, instance: u8, word_offset: u32, dst: &mut [u32]) {
    let end = (word_offset as u64 + dst.len() as u64) * 4;
    assert!(
        end <= DATA_WINDOW_BYTES as u64,
        "data SRAM read of {} words at offset {} leaves the {} byte window",
        dst.len(),
        word_offset,
        DATA_WINDOW_BYTES
    );
    let base = (data_addr(quadrant, instance) + word_offset * 4) as *const u32;
    for (i, word) in dst.iter_mut().enumerate() {
        *word = unsafe { base.add(i).read_volatile() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Addresses cross-checked against generated cnn.c and weights.h in the
    // MaximSDK MAX78002 CNN examples, and against the C_* base constants in
    // ai8x-synthesis izer/tornadocnn.py.

    #[test]
    fn kernel_addresses() {
        // weights.h KERNELS blobs run from 0x51400000 to 0x545e0000.
        assert_eq!(kernel_addr(0, 0), 0x5140_0000);
        assert_eq!(kernel_addr(0, 1), 0x5142_0000);
        assert_eq!(kernel_addr(0, 15), 0x515e_0000);
        assert_eq!(kernel_addr(3, 15), 0x545e_0000);
    }

    #[test]
    fn bias_addresses() {
        // cnn_load_bias writes 0x51180000 through 0x54180000.
        assert_eq!(bias_addr(0), 0x5118_0000);
        assert_eq!(bias_addr(3), 0x5418_0000);
    }

    #[test]
    fn tram_addresses() {
        // C_TRAM_BASE = C_CNN_BASE + 0x200000, processor stride 0x10000.
        assert_eq!(tram_addr(0, 0), 0x5120_0000);
        assert_eq!(tram_addr(0, 15), 0x512f_0000);
        assert_eq!(tram_addr(3, 0), 0x5420_0000);
        // 16 processors of 0x10000 fit between TRAM and the kernel memory.
        assert!(tram_addr(0, 15) + 0x1_0000 <= kernel_addr(0, 0));
    }

    #[test]
    fn data_addresses() {
        // kws20_demo cnn_load_data walks 0x51800000..0x51860000 per quadrant.
        assert_eq!(data_addr(0, 0), 0x5180_0000);
        assert_eq!(data_addr(0, 3), 0x5186_0000);
        // imagenet loads its input at 0x54860000.
        assert_eq!(data_addr(3, 3), 0x5486_0000);
    }

    /// The documented per-processor capacities must sum to the total that
    /// `ai8xize.py` reports.
    #[test]
    fn kernel_capacity_totals_match_datasheet() {
        let kernels: u32 = (0..QUADRANTS)
            .map(|_| {
                (0..PROCESSORS_PER_QUADRANT)
                    .map(kernel_capacity)
                    .sum::<u32>()
            })
            .sum();
        assert_eq!(kernels * 9, TOTAL_KERNEL_BYTES);
    }

    #[test]
    fn data_window_holds_four_processor_slots() {
        assert_eq!(
            DATA_INSTANCE_STRIDE_WORDS * (DATA_INSTANCES_PER_QUADRANT as u32) * 4,
            DATA_WINDOW_BYTES
        );
        // Backed memory is smaller than the address space it sits in.
        assert!(DATA_INSTANCE_WORDS < DATA_INSTANCE_STRIDE_WORDS);
        // 64 processors x 5120 words x 4 bytes is the advertised 1.31MB.
        let total =
            DATA_INSTANCE_WORDS as u64 * (QUADRANTS as u64) * (PROCESSORS_PER_QUADRANT as u64) * 4;
        assert_eq!(total, 1_310_720);
    }

    /// `imagenet` writes 12544 words to one instance in a single call. Any
    /// bound tighter than the window would reject it.
    #[test]
    fn bound_admits_the_largest_shipped_input_load() {
        assert!(12544 * 4 <= DATA_WINDOW_BYTES);
        assert!(12544 > DATA_INSTANCE_WORDS);
    }

    /// Record addresses taken from the `KERNELS` blobs in `weights.h`. Each
    /// record is `(address, length, data...)`, and the address decomposes into
    /// a quadrant, a processor and a byte offset.
    #[test]
    fn kernel_burst_addresses_match_the_weight_blobs() {
        // kws20_demo, the first five records of processor 0.
        assert_eq!(kernel_addr(0, 0) + 0x000, 0x5140_0000);
        assert_eq!(kernel_addr(0, 0) + 0x210, 0x5140_0210);
        assert_eq!(kernel_addr(0, 0) + 0x2b0, 0x5140_02b0);
        assert_eq!(kernel_addr(0, 0) + 0x450, 0x5140_0450);
        assert_eq!(kernel_addr(0, 0) + 0x500, 0x5140_0500);
        // The next processor starts a fresh window.
        assert_eq!(kernel_addr(0, 1) + 0x000, 0x5142_0000);
        assert_eq!(kernel_addr(0, 1) + 0x210, 0x5142_0210);
    }

    /// The window is far larger than any shipped network needs, so the bound
    /// is generous by design. `pascalvoc-retinanetv7_3` has the longest single
    /// burst at 5686 words; `imagenet-riscv` reaches the highest end offset.
    #[test]
    fn bound_admits_the_largest_shipped_weight_bursts() {
        assert!(kernel_burst_fits(0, 5686));
        assert!(kernel_burst_fits(0x5b7c - 4611 * 4, 4611));
        // Roughly a fifth of the window is in use at worst.
        assert!(0x5b7c * 5 < KERNEL_WINDOW_BYTES);
    }

    #[test]
    fn bound_rejects_bursts_that_leave_the_window() {
        assert!(kernel_burst_fits(0, (KERNEL_WINDOW_BYTES / 4) as usize));
        assert!(!kernel_burst_fits(
            0,
            (KERNEL_WINDOW_BYTES / 4) as usize + 1
        ));
        assert!(!kernel_burst_fits(KERNEL_WINDOW_BYTES, 1));
        assert!(kernel_burst_fits(KERNEL_WINDOW_BYTES - 4, 1));
        // The arithmetic must not wrap.
        assert!(!kernel_burst_fits(u32::MAX, 1));
    }
}
