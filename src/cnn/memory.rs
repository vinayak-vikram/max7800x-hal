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
///
/// # Safety
/// `addr` must be a [`kernel_addr`] for a quadrant the caller owns.
#[inline]
pub unsafe fn arm_kernel_ptr(addr: u32) {
    ((addr | 1) as *mut u8).write_volatile(0x01);
}

/// Whether `entries` bias values fit in one quadrant's bias memory.
#[inline]
pub const fn bias_fits(entries: usize) -> bool {
    entries as u64 <= BIAS_ENTRIES as u64
}

/// Write bias values into a quadrant's bias memory.
pub fn write_bias(quadrant: u8, data: &[u8]) {
    assert!(
        bias_fits(data.len()),
        "{} bias entries exceed the {} the quadrant holds",
        data.len(),
        BIAS_ENTRIES
    );
    let base = bias_addr(quadrant) as *mut u32;
    for (i, value) in data.iter().enumerate() {
        // SAFETY: bounded above, inside a quadrant the caller owns.
        unsafe { base.add(i).write_volatile(*value as u32) };
    }
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
