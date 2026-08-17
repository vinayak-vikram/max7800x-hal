//! # CNN Layer Register Values
//!

use super::regs::LayerReg;

macro_rules! register {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name(u32);

        impl $name {
            /// A zeroed register.
            #[inline]
            pub const fn new() -> Self {
                Self(0)
            }

            #[inline]
            pub const fn from_bits(bits: u32) -> Self {
                Self(bits)
            }

            #[inline]
            pub const fn bits(self) -> u32 {
                self.0
            }
        }
    };
}

/// Declares a getter for a multi-bit field at `[lsb + width - 1:lsb]`.
macro_rules! field {
    ($(#[$attr:meta])* $get:ident, $lsb:expr, $width:expr) => {
        $(#[$attr])*
        #[inline]
        pub const fn $get(self) -> u32 {
            (self.0 >> $lsb) & (u32::MAX >> (32 - $width))
        }
    };
}

/// Declares a getter for a single-bit field.
macro_rules! flag {
    ($(#[$attr:meta])* $get:ident, $bit:expr) => {
        $(#[$attr])*
        #[inline]
        pub const fn $get(self) -> bool {
            self.0 & (1 << $bit) != 0
        }
    };
}

register! {
    /// Next layer
    Nxtlyr
}

impl Nxtlyr {
    field!(next, 0, 7);
    flag!(link_en, 7);
    flag!(stop, 8);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x000001ff;
}

impl core::fmt::Debug for Nxtlyr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("Nxtlyr");
        if self.link_en() {
            s.field("next", &self.next());
        }
        s.field("link_en", &self.link_en())
            .field("stop", &self.stop())
            .finish()
    }
}

register! {
    /// Row count
    Rcnt
}

register! {
    /// Column count
    Ccnt
}

macro_rules! count_register {
    ($name:ident) => {
        impl $name {
            field!(cnt, 0, 11);
            field!(pad_cnt, 13, 2);
            flag!(pad_ena, 15);
            field!(diff, 16, 16);
            /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
            pub const DECLARED_BITS: u32 = 0xffffe7ff;
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let pad = if self.pad_ena() {
                    self.pad_cnt() + 1
                } else {
                    0
                };
                f.debug_struct(stringify!($name))
                    .field("cnt", &self.cnt())
                    .field("pad", &pad)
                    .field("diff", &self.diff())
                    .finish()
            }
        }
    };
}

count_register!(Rcnt);
count_register!(Ccnt);

register! {
    /// Pooling rows
    Prcnt
}

register! {
    /// Pooling columns
    Pccnt
}

macro_rules! pool_register {
    ($name:ident) => {
        impl $name {
            field!(pool_cnt, 0, 4);
            field!(pool_inc, 4, 4);
            /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
            pub const DECLARED_BITS: u32 = 0x000000ff;
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("pool_cnt", &self.pool_cnt())
                    .field("pool_inc", &self.pool_inc())
                    .finish()
            }
        }
    };
}

pool_register!(Prcnt);
pool_register!(Pccnt);

register! {
    /// Pooling and multi-pass stride
    Stride
}

impl Stride {
    field!(stride, 0, 4);
    field!(mp_stride, 4, 28);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0xffffffff;
}

impl core::fmt::Debug for Stride {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stride")
            .field("stride", &self.stride())
            .field("mp_stride", &self.mp_stride())
            .finish()
    }
}

register! {
    /// Data SRAM write pointer
    WptrBase
}

impl WptrBase {
    field!(offset, 0, 13);
    field!(instance, 13, 2);
    field!(group, 15, 6);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x001fffff;
}

impl core::fmt::Debug for WptrBase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WptrBase")
            .field("offset", &self.offset())
            .field("instance", &self.instance())
            .field("group", &self.group())
            .finish()
    }
}

/// Declares a register whose whole word is one unstructured value
macro_rules! value_register {
    ($(#[$attr:meta])* $name:ident) => {
        register! { $(#[$attr])* $name }

        impl $name {
            /// The whole word is one value, so no bit is reserved
            pub const DECLARED_BITS: u32 = u32::MAX;
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({:#x})", stringify!($name), self.0)
            }
        }
    };
}

value_register! {
    /// Write pointer time slot offset
    WptrToffs
}

value_register! {
    /// Write pointer mask offset
    WptrMoffs
}

value_register! {
    /// Write pointer multi-pass channel offset
    WptrChoffs
}

value_register! {
    /// Data SRAM read pointer
    RptrBase
}

register! {
    /// Layer control
    Lctl
}

impl Lctl {
    flag!(unnamed5, 5);
    flag!(chw, 6);
    flag!(pool_ena, 7);
    flag!(maxpool, 8);
    flag!(relu, 9);
    flag!(global_wptr, 11);
    field!(siena, 12, 4);
    flag!(wide_out, 16);
    flag!(rd_ahead, 17);
    field!(cprime_max, 18, 4);
    field!(rprime_max, 22, 4);
    field!(shift_cnt, 26, 4);
    flag!(dw_bcast, 29);
    flag!(bypass, 30);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x7ffffbe0;
}

impl core::fmt::Debug for Lctl {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("Lctl");
        s.field("unnamed5", &self.unnamed5())
            .field("chw", &self.chw())
            .field("pool_ena", &self.pool_ena())
            .field("maxpool", &self.maxpool())
            .field("relu", &self.relu())
            .field("global_wptr", &self.global_wptr())
            .field("siena", &self.siena())
            .field("wide_out", &self.wide_out())
            .field("rd_ahead", &self.rd_ahead())
            .field("kernel", &(self.rprime_max() + 1, self.cprime_max() + 1));
        if self.rd_ahead() {
            s.field("shift_cnt", &self.shift_cnt());
        } else {
            s.field("dw_bcast", &self.dw_bcast());
        }
        s.field("bypass", &self.bypass()).finish()
    }
}

register! {
    /// Layer control 2
    Lctl2
}

impl Lctl2 {
    field!(maxpass, 0, 4);
    field!(wptr_inc, 4, 8);
    field!(xpch_max, 12, 9);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x001fffff;
}

impl core::fmt::Debug for Lctl2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lctl2")
            .field("maxpass", &self.maxpass())
            .field("wptr_inc", &self.wptr_inc())
            .field("xpch_max", &self.xpch_max())
            .finish()
    }
}

register! {
    /// 1D convolution & element-wise configuration
    Oned
}

impl Oned {
    field!(tscnt_max, 0, 4);
    field!(oned_sad, 4, 4);
    field!(oned_width, 8, 4);
    flag!(oned_ena, 12);
    flag!(elt_ena, 13);
    field!(elt_fn, 14, 2);
    flag!(pool_first, 16);
    flag!(elt_conv, 17);
    field!(operands, 18, 4);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x003fffff;
}

impl core::fmt::Debug for Oned {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Oned")
            .field("tscnt_max", &self.tscnt_max())
            .field("oned_sad", &self.oned_sad())
            .field("oned_width", &self.oned_width())
            .field("oned_ena", &self.oned_ena())
            .field("elt_ena", &self.elt_ena())
            .field("elt_fn", &self.elt_fn())
            .field("pool_first", &self.pool_first())
            .field("elt_conv", &self.elt_conv())
            .field("operands", &(self.operands() + 1))
            .finish()
    }
}

register! {
    /// Last mask memory word
    /// Not written for passthrough layers
    Mcnt1
}

impl Mcnt1 {
    field!(mexp_max, 0, 19);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x0007ffff;
}

impl core::fmt::Debug for Mcnt1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mcnt1")
            .field("mexp_max", &self.mexp_max())
            .finish()
    }
}

register! {
    /// First mask memory word
    Mcnt2
}

impl Mcnt2 {
    field!(mexp_sad, 0, 19);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x0007ffff;
}

impl core::fmt::Debug for Mcnt2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mcnt2")
            .field("mexp_sad", &self.mexp_sad())
            .finish()
    }
}

register! {
    /// Output channel count minus one
    Ochan
}

impl Ochan {
    field!(ochan, 0, 32);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0xffffffff;
}

impl core::fmt::Debug for Ochan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ochan")
            .field("channels", &self.ochan().saturating_add(1))
            .finish()
    }
}

register! {
    /// TRAM pointer
    Tptr
}

impl Tptr {
    field!(tptr_max, 0, 16);
    field!(tptr_sad, 16, 16);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0xffffffff;
}

impl core::fmt::Debug for Tptr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tptr")
            .field("tptr_max", &self.tptr_max())
            .field("tptr_sad", &self.tptr_sad())
            .finish()
    }
}

register! {
    /// Processor and mask enables
    Ena
}

impl Ena {
    field!(proc_ena, 0, 16);
    field!(mask_ena, 16, 16);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0xffffffff;
}

impl core::fmt::Debug for Ena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ena")
            .field("proc_ena", &format_args!("{:#06x}", self.proc_ena()))
            .field("mask_ena", &format_args!("{:#06x}", self.mask_ena()))
            .finish()
    }
}

register! {
    /// Post processing.
    Post
}

impl Post {
    field!(bias_addr, 0, 12);
    flag!(bias_en, 12);
    field!(scale_mag, 13, 4);
    flag!(scale_dir, 17);
    field!(xpmp_cnt, 18, 4);
    field!(wscale, 22, 2);
    flag!(ts_ena, 24);
    flag!(onexone_ena, 25);
    flag!(act_abs, 26);
    flag!(flatten_ena, 27);
    flag!(xpose_ena, 28);
    flag!(calcx4, 29);
    flag!(dw_ena, 30);
    flag!(tcalc, 31);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0xffffffff;

    #[inline]
    pub const fn shift_dir(self) -> ShiftDir {
        if self.scale_dir() {
            ShiftDir::Right
        } else {
            ShiftDir::Left
        }
    }

    /// Weight-width compensation.
    #[inline]
    pub const fn weight_scale(self) -> WeightScale {
        WeightScale::from_code(self.wscale())
    }

    #[inline]
    pub const fn output_shift(self) -> i32 {
        let mag = self.scale_mag() as i32;
        if self.scale_dir() {
            -mag
        } else {
            mag
        }
    }
}

impl core::fmt::Debug for Post {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Post")
            .field("bias_addr", &self.bias_addr())
            .field("bias_en", &self.bias_en())
            .field("output_shift", &self.output_shift())
            .field("xpmp_cnt", &self.xpmp_cnt())
            .field("weight_scale", &self.weight_scale())
            .field("ts_ena", &self.ts_ena())
            .field("onexone_ena", &self.onexone_ena())
            .field("act_abs", &self.act_abs())
            .field("flatten_ena", &self.flatten_ena())
            .field("xpose_ena", &self.xpose_ena())
            .field("calcx4", &self.calcx4())
            .field("dw_ena", &self.dw_ena())
            .field("tcalc", &self.tcalc())
            .finish()
    }
}

/// Pooling mode, `LCTL.maxpool`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PoolMode {
    Avg,
    Max,
}

impl Lctl {
    #[inline]
    pub const fn pool_mode(self) -> PoolMode {
        if self.maxpool() {
            PoolMode::Max
        } else {
            PoolMode::Avg
        }
    }
}

/// Element-wise function, `ONED.elt_fn`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum EltwiseFn {
    Sub = 0b00,
    Add = 0b01,
    Or = 0b10,
    Xor = 0b11,
}

impl EltwiseFn {
    #[inline]
    pub const fn from_code(code: u32) -> Self {
        match code & 0b11 {
            0b00 => Self::Sub,
            0b01 => Self::Add,
            0b10 => Self::Or,
            _ => Self::Xor,
        }
    }
}

impl Oned {
    #[inline]
    pub const fn eltwise_fn(self) -> EltwiseFn {
        EltwiseFn::from_code(self.elt_fn())
    }
}

/// Weight-width compensation, `POST.wscale`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum WeightScale {
    /// Eight-bit weights, or a bypass layer. No compensation.
    Bits8 = 0,
    /// One-bit weights, including binary.
    Bits1 = 1,
    Bits2 = 2,
    Bits4 = 3,
}

impl WeightScale {
    #[inline]
    pub const fn from_code(code: u32) -> Self {
        match code & 0b11 {
            0 => Self::Bits8,
            1 => Self::Bits1,
            2 => Self::Bits2,
            _ => Self::Bits4,
        }
    }
}

/// Output shift direction, `POST.scale_dir`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum ShiftDir {
    Left = 0,
    Right = 1,
}

/// Activation, which the hardware splits across two registers: ReLU is
/// `LCTL.relu` and absolute value is `POST.act_abs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    None,
    Relu,
    Abs,
}

impl Activation {
    /// Read the activation out of the two registers that carry it.
    #[inline]
    pub const fn decode(lctl: Lctl, post: Post) -> Option<Self> {
        match (lctl.relu(), post.act_abs()) {
            (false, false) => Some(Self::None),
            (true, false) => Some(Self::Relu),
            (false, true) => Some(Self::Abs),
            (true, true) => None, //should never happen, UB basically
        }
    }
}

register! {
    /// Stream processing start
    Stream1
}

impl Stream1 {
    field!(isval, 0, 15);
    flag!(fifo_go, 25);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x02007fff;
}

impl core::fmt::Debug for Stream1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream1")
            .field("isval", &self.isval())
            .field("fifo_go", &self.fifo_go())
            .finish()
    }
}

register! {
    /// Stream processing delta
    Stream2
}

impl Stream2 {
    field!(invol, 0, 4);
    field!(dsval1, 4, 5);
    field!(dsval2, 16, 14);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x3fff01ff;
}

impl core::fmt::Debug for Stream2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stream2")
            .field("invol", &self.invol())
            .field("dsval1", &self.dsval1())
            .field("dsval2", &self.dsval2())
            .finish()
    }
}

register! {
    /// Ring buffer size
    Fmax
}

impl Fmax {
    field!(fbuf_max, 0, 18);
    /// Bits covered by a field above; pinned by `declared_bits_match_the_getters`
    pub const DECLARED_BITS: u32 = 0x0003ffff;
}

impl core::fmt::Debug for Fmax {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fmax")
            .field("fbuf_max", &self.fbuf_max())
            .finish()
    }
}

/// its beautiful 🥹
pub trait LayerRegister: Copy + core::fmt::Debug {
    const REG: LayerReg;
    /// Bits this register's fields cover; the rest must read back zero
    const DECLARED: u32;

    fn from_bits(bits: u32) -> Self;
    fn bits(self) -> u32;

    /// Set bits that belong to no field, which means a miscompiled word
    #[inline]
    fn reserved(self) -> u32 {
        self.bits() & !Self::DECLARED
    }
}

macro_rules! layer_registers {
    ($($ty:ident => $reg:ident),* $(,)?) => {
        $(
            impl LayerRegister for $ty {
                const REG: LayerReg = LayerReg::$reg;
                const DECLARED: u32 = $ty::DECLARED_BITS;

                #[inline]
                fn from_bits(bits: u32) -> Self {
                    Self(bits)
                }

                #[inline]
                fn bits(self) -> u32 {
                    self.0
                }
            }
        )*

        /// Every register that has a value type, in emit order.
        pub const ALL_TYPED_REGS: [LayerReg; 20] = [$(LayerReg::$reg),*];

        /// Set bits of `bits` that belong to no field of `reg`
        pub fn reserved_bits(reg: LayerReg, bits: u32) -> u32 {
            match reg {
                $(LayerReg::$reg => $ty::from_bits(bits).reserved()),*
            }
        }

        // for debugging
        pub fn with_decoded<T>(
            reg: LayerReg,
            bits: u32,
            f: impl FnOnce(&dyn core::fmt::Debug) -> T,
        ) -> T {
            match reg {
                $(LayerReg::$reg => f(&$ty(bits))),*
            }
        }
    };
}

layer_registers! {
    Nxtlyr => Next,
    Rcnt => Rows,
    Ccnt => Cols,
    Prcnt => PoolRows,
    Pccnt => PoolCols,
    Stride => Stride,
    WptrBase => Wptr,
    WptrToffs => WptrTs,
    WptrMoffs => WptrMask,
    WptrChoffs => WptrMp,
    RptrBase => Rptr,
    Lctl => Lctl,
    Lctl2 => Lctl2,
    Mcnt1 => Mcnt,
    Mcnt2 => Moffs,
    Ochan => Ochan,
    Oned => Oned,
    Tptr => Tptr,
    Post => Post,
    Ena => En,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `field!` and `flag!` macros are six lines of shift and mask shared
    /// by all sixty-seven fields, so they are worth testing once rather than
    /// once per register. `Rcnt` covers a field at offset 0, one abutting its
    /// neighbour, a flag, and a field running to bit 31.
    #[test]
    fn getters_shift_and_mask() {
        let all = Rcnt::from_bits(u32::MAX);
        assert_eq!(all.cnt(), 0x7ff);
        assert_eq!(all.pad_cnt(), 0b11);
        assert!(all.pad_ena());
        assert_eq!(all.diff(), 0xffff);

        // Each field sees only its own bits.
        assert_eq!(Rcnt::from_bits(0xffff_0000).cnt(), 0);
        assert_eq!(Rcnt::from_bits(0x0000_ffff).diff(), 0);
        assert_eq!(Rcnt::from_bits(!(1 << 15)).pad_ena(), false);

        // A full-width field is not truncated, and a zero word reads as zero.
        assert_eq!(Ochan::from_bits(u32::MAX).ochan(), u32::MAX);
        assert_eq!(Rcnt::from_bits(0).diff(), 0);
    }

    /// mobilefacenet-112 layer 0: 112x112 input, pad 1, no pooling.
    #[test]
    fn mobilefacenet_layer0() {
        let rows = Rcnt::from_bits(0x0001_806f);
        assert_eq!(rows.cnt(), 111);
        assert!(rows.pad_ena());
        assert_eq!(rows.pad_cnt() + 1, 1);
        assert_eq!(rows.diff(), 1);

        let stride = Stride::from_bits(0x0000_0010);
        assert_eq!(stride.stride() + 1, 1);
        assert_eq!(stride.mp_stride(), 1);

        let wptr = WptrBase::from_bits(0x0000_2000);
        assert_eq!(wptr.offset(), 0);
        assert_eq!(wptr.instance(), 1);
        assert_eq!(wptr.group(), 0);
    }

    /// mobilefacenet-112 layer 1: 2x2 pool, stride 2.
    #[test]
    fn mobilefacenet_layer1() {
        let rows = Rcnt::from_bits(0x0072_806e);
        assert_eq!(rows.cnt(), 110);
        assert!(rows.pad_ena());
        assert_eq!(rows.diff(), 114);

        let cols = Ccnt::from_bits(0x0002_806e);
        assert_eq!(cols.cnt(), 110);
        assert_eq!(cols.diff(), 2);

        let pool = Prcnt::from_bits(0x0000_0001);
        assert_eq!(pool.pool_cnt(), 1);
        assert_eq!(pool.pool_inc(), 0);

        let stride = Stride::from_bits(0x0000_0021);
        assert_eq!(stride.stride() + 1, 2);
        assert_eq!(stride.mp_stride(), 2);
    }

    /// kws20_demo layer 0: Conv1d, so the column count is degenerate.
    #[test]
    fn kws20_layer0() {
        let rows = Rcnt::from_bits(0x0002_007f);
        assert_eq!(rows.cnt(), 127);
        assert!(!rows.pad_ena());
        assert_eq!(rows.diff(), 2);

        let cols = Ccnt::from_bits(0x0001_0000);
        assert_eq!(cols.cnt(), 0);
        assert_eq!(cols.diff(), 1);
    }

    /// The passthrough layer izer inserts before an average pool that follows
    /// an element-wise op writes a fixed set of words.
    #[test]
    fn inserted_passthrough_layer() {
        let rows = Rcnt::from_bits(0x0001_0000);
        assert_eq!(rows.cnt(), 0);
        assert_eq!(rows.diff(), 1);

        let stride = Stride::from_bits(0x0000_0010);
        assert_eq!(stride.stride(), 0);
        assert_eq!(stride.mp_stride(), 1);
    }

    #[test]
    fn nxtlyr_encodes_link_and_stop() {
        let sequential = Nxtlyr::new();
        assert!(!sequential.link_en());
        assert!(!sequential.stop());
        assert_eq!(sequential.bits(), 0);

        // Link-layer mode on a non-final layer: LINK_EN | (ll + 1).
        let linked = Nxtlyr::from_bits(0x87);
        assert_eq!(linked.bits(), 0x87);
        assert_eq!(linked.next(), 7);

        // Link-layer mode on the final layer.
        let stopped = Nxtlyr::from_bits(0x100);
        assert_eq!(stopped.bits(), 0x100);

        // The snoop-loop override force-writes a link back to layer 0.
        let loop_back = Nxtlyr::from_bits(0x80);
        assert!(loop_back.link_en());
        assert_eq!(loop_back.next(), 0);
    }

    /// kws20_demo layer 0: Conv1d, master quadrant, in a network with no bias.
    #[test]
    fn kws20_layer0_control() {
        let master = Lctl::from_bits(0x0000_eb20);
        assert!(master.unnamed5());
        assert!(master.relu());
        assert!(master.global_wptr());
        assert_eq!(master.siena(), 0b1110);
        assert!(!master.rd_ahead());

        // The same layer on a non-master quadrant differs only in SIENA.
        let slave = Lctl::from_bits(0x0000_0b20);
        assert_eq!(slave.siena(), 0);
        assert_eq!(slave.bits(), master.bits() & !(0xf << 12));

        let l2 = Lctl2::from_bits(0x0019_8011);
        assert_eq!(l2.maxpass(), 1);
        assert_eq!(l2.wptr_inc(), 1);
        assert_eq!(l2.xpch_max(), 408);

        let oned = Oned::from_bits(0x0000_1100);
        assert!(oned.oned_ena());
        assert_eq!(oned.oned_width(), 1);
        assert!(!oned.elt_ena());
        assert_eq!(oned.operands() + 1, 1);

        assert_eq!(Ochan::from_bits(0x0000_00cf).ochan() + 1, 208);
        assert_eq!(Mcnt1::from_bits(0x0000_0678).mexp_max(), 0x678);

        let ena = Ena::from_bits(0xffff_ffff);
        assert_eq!(ena.proc_ena(), 0xffff);
        assert_eq!(ena.mask_ena(), ena.proc_ena());
    }

    /// cifar-100-effnet2 layer 15 is the only shipped layer that sets LCTL bit
    /// 29. It is a genuine depthwise broadcast, not a `shift_cnt` spill:
    /// `rd_ahead` is clear, so `shift_cnt` is not written at all.
    #[test]
    fn lctl_bit29_is_depthwise_broadcast_in_practice() {
        let l = Lctl::from_bits(0x2088_0b20);
        assert!(!l.rd_ahead());
        assert!(l.dw_bcast());
        assert_eq!(l.cprime_max() + 1, 3);
        assert_eq!(l.rprime_max() + 1, 3);
    }

    /// The unguarded case `validate()` must reject: read-ahead without tcalc
    /// and a large input expansion sets bit 29 through `shift_cnt` alone.
    #[test]
    fn lctl_shift_cnt_can_forge_broadcast() {
        let l = Lctl::from_bits((1 << 17) | (8 << 26));
        assert!(l.dw_bcast());
        assert_eq!(l.shift_cnt(), 8);
    }

    /// The inserted average-pool reset layer writes a fixed control word.
    #[test]
    fn inserted_passthrough_layer_control() {
        let l = Lctl::from_bits(0x920);
        assert!(l.unnamed5());
        assert!(l.maxpool());
        assert!(l.global_wptr());
        assert!(!l.pool_ena());
        assert_eq!(l.siena(), 0);

        let ena = Ena::from_bits(1);
        assert_eq!(ena.proc_ena(), 1);
        assert_eq!(ena.mask_ena(), 0);
    }

    /// Element-wise layers encode the operand count minus one.
    #[test]
    fn eltwise_operands() {
        let o = Oned::from_bits(0x0004_6003);
        assert!(o.elt_ena());
        assert!(!o.oned_ena());
        assert_eq!(o.elt_fn(), 1);
        assert_eq!(o.operands() + 1, 2);
        assert!(!o.pool_first());

        // Same layer with pooling ahead of the element-wise op.
        let pooled = Oned::from_bits(0x0005_6003);
        assert!(pooled.pool_first());
        assert_eq!(pooled.bits(), o.bits() | (1 << 16));
    }

    const POSTS: [u32; 16] = [
        0x0000_1000,
        0x0000_113e,
        0x0000_15b0,
        0x0000_2000,
        0x0000_3100,
        0x0000_5430,
        0x0000_73f0,
        0x0000_8000,
        0x0002_2000,
        0x0002_34c4,
        0x0002_5000,
        0x0002_73b0,
        0x0100_0000,
        0x0300_0000,
        0x1000_1200,
        0x4100_d004,
    ];

    /// The field the plan singles out as most error-prone: five bits of signed
    /// magnitude, not two's complement.
    #[test]
    fn output_shift_is_signed_magnitude() {
        // The worked example from the specification.
        let p = Post::from_bits((3 << 13) | (1 << 17));
        assert_eq!(p.scale_mag(), 3);
        assert!(p.scale_dir());
        assert_eq!(p.bits() >> 13, 0b1_0011);
        assert_ne!(p.bits() >> 13, 0b1_1101, "encoded as two's complement");
        assert_eq!(p.output_shift(), -3);

        for shift in -15..=15i32 {
            let raw = (shift.unsigned_abs() << 13) | ((shift < 0) as u32) << 17;
            assert_eq!(Post::from_bits(raw).output_shift(), shift, "shift {shift}");
        }

        // Every scale actually shipped, decoded from the golden words.
        assert_eq!(Post::from_bits(0x0000_8000).output_shift(), 4);
        assert_eq!(Post::from_bits(0x0000_73f0).output_shift(), 3);
        assert_eq!(Post::from_bits(0x0002_2000).output_shift(), -1);
        assert_eq!(Post::from_bits(0x0002_5000).output_shift(), -2);
        assert_eq!(Post::from_bits(0x0002_73b0).output_shift(), -3);
        assert_eq!(Post::from_bits(0x4100_d004).output_shift(), 6);
    }

    #[test]
    fn shift_direction_matches_the_raw_bit() {
        assert_eq!(Post::from_bits(2 << 13).shift_dir(), ShiftDir::Left);
        assert_eq!(
            Post::from_bits((2 << 13) | (1 << 17)).shift_dir(),
            ShiftDir::Right
        );
        assert!(Post::from_bits(1 << 17).scale_dir());
    }

    /// cifar-100-effnet2 layer 15: depthwise, so `dw_ena` and `ts_ena` are
    /// written together.
    #[test]
    fn effnet2_layer15_post() {
        let p = Post::from_bits(0x4100_3000);
        assert!(p.dw_ena());
        assert!(p.ts_ena(), "dw_ena is always written with ts_ena");
        assert!(p.bias_en());
        assert_eq!(p.bias_addr(), 0);
        assert_eq!(p.output_shift(), 1);
        assert_eq!(p.weight_scale(), WeightScale::Bits8);
    }

    /// The inserted average-pool reset layer writes `POST = 0x0300_0000`.
    #[test]
    fn inserted_passthrough_layer_post() {
        let p = Post::from_bits(0x0300_0000);
        assert!(p.ts_ena());
        assert!(p.onexone_ena());
        assert!(!p.bias_en());
        assert_eq!(p.output_shift(), 0, "passthrough must not shift");
    }

    #[test]
    fn weight_scale_codes_match_the_generator() {
        assert_eq!(WeightScale::from_code(0), WeightScale::Bits8);
        assert_eq!(WeightScale::from_code(1), WeightScale::Bits1);
        assert_eq!(WeightScale::from_code(2), WeightScale::Bits2);
        assert_eq!(WeightScale::from_code(3), WeightScale::Bits4);
        for s in [
            WeightScale::Bits8,
            WeightScale::Bits1,
            WeightScale::Bits2,
            WeightScale::Bits4,
        ] {
            assert_eq!(Post::from_bits((s as u32) << 22).weight_scale(), s);
        }
        // Every shipped network uses 8-bit weights.
        for bits in POSTS {
            assert_eq!(Post::from_bits(bits).weight_scale(), WeightScale::Bits8);
        }
    }

    #[test]
    fn eltwise_codes_match_the_generator() {
        assert_eq!(EltwiseFn::from_code(0b00), EltwiseFn::Sub);
        assert_eq!(EltwiseFn::from_code(0b01), EltwiseFn::Add);
        assert_eq!(EltwiseFn::from_code(0b10), EltwiseFn::Or);
        assert_eq!(EltwiseFn::from_code(0b11), EltwiseFn::Xor);
        // The shipped element-wise layers are adds.
        assert_eq!(Oned::from_bits(0x0004_6003).eltwise_fn(), EltwiseFn::Add);
        for f in [
            EltwiseFn::Sub,
            EltwiseFn::Add,
            EltwiseFn::Or,
            EltwiseFn::Xor,
        ] {
            assert_eq!(Oned::from_bits((f as u32) << 14).eltwise_fn(), f);
        }
    }

    #[test]
    fn pool_mode_follows_maxpool() {
        assert_eq!(Lctl::from_bits(0x920).pool_mode(), PoolMode::Max);
        assert_eq!(Lctl::new().pool_mode(), PoolMode::Avg);
        assert_eq!(Lctl::from_bits(1 << 8).pool_mode(), PoolMode::Max);
    }

    /// Activation spans two registers, so it round-trips through both.
    #[test]
    fn activation_spans_lctl_and_post() {
        let lctl = Lctl::from_bits(0x0000_eb20);
        let post = Post::from_bits(0x0000_2000);
        assert_eq!(Activation::decode(lctl, post), Some(Activation::Relu));

        for (l, p, act) in [
            (0, 0, Activation::None),
            (1 << 9, 0, Activation::Relu),
            (0, 1 << 26, Activation::Abs),
        ] {
            let decoded = Activation::decode(Lctl::from_bits(l), Post::from_bits(p));
            assert_eq!(decoded, Some(act));
        }

        // Both bits set is not a valid encoding.
        let both = Lctl::from_bits(1 << 9);
        let abs = Post::from_bits(1 << 26);
        assert_eq!(Activation::decode(both, abs), None);
    }

    fn assert_reg<R: LayerRegister>(expected: LayerReg, bits: u32) {
        assert_eq!(R::REG, expected);
        assert_eq!(<R as LayerRegister>::from_bits(bits).bits(), bits);
    }

    /// Pins the value-type to register-slot mapping one type at a time. The
    /// table in `layer_registers!` is the only place it is written down, so a
    /// swapped pair there would otherwise go unnoticed.
    #[test]
    fn every_type_maps_to_its_own_slot() {
        assert_reg::<Nxtlyr>(LayerReg::Next, 0x87);
        assert_reg::<Rcnt>(LayerReg::Rows, 0x0002_007f);
        assert_reg::<Ccnt>(LayerReg::Cols, 0x0001_0000);
        assert_reg::<Oned>(LayerReg::Oned, 0x0000_1100);
        assert_reg::<Prcnt>(LayerReg::PoolRows, 0x1);
        assert_reg::<Pccnt>(LayerReg::PoolCols, 0x1);
        assert_reg::<Stride>(LayerReg::Stride, 0x20);
        assert_reg::<WptrBase>(LayerReg::Wptr, 0x800);
        assert_reg::<WptrToffs>(LayerReg::WptrTs, 0x1);
        assert_reg::<WptrMoffs>(LayerReg::WptrMask, 0x8000);
        assert_reg::<WptrChoffs>(LayerReg::WptrMp, 0x1);
        assert_reg::<RptrBase>(LayerReg::Rptr, 0x2000);
        assert_reg::<Lctl>(LayerReg::Lctl, 0x0000_eb20);
        assert_reg::<Lctl2>(LayerReg::Lctl2, 0x0019_8011);
        assert_reg::<Mcnt1>(LayerReg::Mcnt, 0x678);
        assert_reg::<Mcnt2>(LayerReg::Moffs, 0x1200);
        assert_reg::<Ochan>(LayerReg::Ochan, 0xcf);
        assert_reg::<Tptr>(LayerReg::Tptr, 0x0140_027f);
        assert_reg::<Ena>(LayerReg::En, 0xffff_ffff);
        assert_reg::<Post>(LayerReg::Post, 0x0000_3000);
    }

    #[test]
    fn typed_registers_cover_every_slot_once() {
        assert_eq!(ALL_TYPED_REGS.len(), crate::cnn::regs::ALL_LAYER_REGS.len());
        for expected in crate::cnn::regs::ALL_LAYER_REGS {
            let count = ALL_TYPED_REGS.iter().filter(|r| **r == expected).count();
            assert_eq!(count, 1, "{expected:?} has {count} value types");
        }
    }

    /// The emit order is a property of the network, not of this table, but
    /// keeping the table in emit order makes a generic emit trivial.
    #[test]
    fn typed_registers_are_in_emit_order() {
        assert_eq!(ALL_TYPED_REGS, super::super::EMIT_ORDER);
    }

    /// Every `DECLARED_BITS` constant is a hand-written literal, so this pins
    /// each one to the fields actually declared above it. A bit is declared
    /// exactly when toggling it changes what some getter reports, which the
    /// `Debug` impl renders. Catches a mask that drifts after a field is
    /// added, moved or widened, in either direction.
    ///
    /// Toggled against both an all-zero and an all-ones word because some
    /// `Debug` impls are conditional: `Nxtlyr` renders `next` only when
    /// `link_en` is set, so probing up from zero alone would miss it.
    #[test]
    fn declared_bits_match_the_getters() {
        for reg in ALL_TYPED_REGS {
            let (zero, ones) = (rendered(reg, 0), rendered(reg, u32::MAX));
            for bit in 0..32 {
                let up = rendered(reg, 1 << bit);
                let down = rendered(reg, u32::MAX ^ (1 << bit));
                let visible = up.as_str() != zero.as_str() || down.as_str() != ones.as_str();
                let declared = reserved_bits(reg, 1 << bit) == 0;
                assert_eq!(
                    visible,
                    declared,
                    "{reg:?} bit {bit} is {} but reads back {}",
                    if declared { "declared" } else { "reserved" },
                    if visible { "visible" } else { "as zero" }
                );
            }
        }
    }

    /// A register whose fields tile the whole word can have nothing reserved.
    #[test]
    fn full_width_registers_reserve_nothing() {
        for reg in [
            LayerReg::Stride,
            LayerReg::WptrTs,
            LayerReg::WptrMask,
            LayerReg::WptrMp,
            LayerReg::Rptr,
            LayerReg::Ochan,
            LayerReg::Tptr,
            LayerReg::En,
            LayerReg::Post,
        ] {
            assert_eq!(reserved_bits(reg, u32::MAX), 0, "{reg:?}");
        }
    }

    /// The registers where the check can actually fire, with the reserved
    /// windows spelled out.
    #[test]
    fn reserved_windows_are_where_expected() {
        for (reg, bits) in [
            (LayerReg::Next, 0xffff_fe00u32),
            (LayerReg::Rows, 0x0000_1800),
            (LayerReg::Cols, 0x0000_1800),
            (LayerReg::Oned, 0xffc0_0000),
            (LayerReg::PoolRows, 0xffff_ff00),
            (LayerReg::PoolCols, 0xffff_ff00),
            (LayerReg::Wptr, 0xffe0_0000),
            (LayerReg::Lctl, 0x8000_041f),
            (LayerReg::Lctl2, 0xffe0_0000),
            (LayerReg::Mcnt, 0xfff8_0000),
            (LayerReg::Moffs, 0xfff8_0000),
        ] {
            assert_eq!(reserved_bits(reg, u32::MAX), bits, "{reg:?}");
            assert_eq!(reserved_bits(reg, !bits), 0, "{reg:?}");
        }
    }

    /// A `core::fmt` sink, so the decoder can be checked without `std`.
    struct Buf {
        data: [u8; 512],
        len: usize,
    }

    impl Buf {
        const fn new() -> Self {
            Self {
                data: [0; 512],
                len: 0,
            }
        }

        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.data[..self.len]).unwrap()
        }
    }

    impl core::fmt::Write for Buf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let n = bytes.len().min(self.data.len() - self.len);
            self.data[self.len..self.len + n].copy_from_slice(&bytes[..n]);
            self.len += n;
            Ok(())
        }
    }

    fn rendered(reg: LayerReg, bits: u32) -> Buf {
        use core::fmt::Write;
        let mut buf = Buf::new();
        with_decoded(reg, bits, |v| write!(buf, "{v:?}").unwrap());
        buf
    }

    #[test]
    fn decoder_visits_every_register() {
        let mut visited = 0;
        for reg in crate::cnn::regs::ALL_LAYER_REGS {
            with_decoded(reg, 0, |_| visited += 1);
        }
        assert_eq!(visited, 20);
    }

    /// The decoder must produce the register's own type, not a same-shaped
    /// neighbour. `Rcnt` and `Ccnt` have identical layouts, so only the name
    /// distinguishes them.
    #[test]
    fn decoder_names_the_right_type() {
        assert!(rendered(LayerReg::Rows, 0x0002_007f)
            .as_str()
            .starts_with("Rcnt"));
        assert!(rendered(LayerReg::Cols, 0x0001_0000)
            .as_str()
            .starts_with("Ccnt"));
        assert!(rendered(LayerReg::PoolRows, 1)
            .as_str()
            .starts_with("Prcnt"));
        assert!(rendered(LayerReg::PoolCols, 1)
            .as_str()
            .starts_with("Pccnt"));
        assert!(rendered(LayerReg::Mcnt, 0x678)
            .as_str()
            .starts_with("Mcnt1"));
        assert!(rendered(LayerReg::Moffs, 0x1200)
            .as_str()
            .starts_with("Mcnt2"));
        assert!(rendered(LayerReg::En, 0xffff_ffff)
            .as_str()
            .starts_with("Ena"));
    }

    /// The payoff the plan is after: a dumped word reads as fields, not hex.
    #[test]
    fn decoder_output_is_decoded_not_hex() {
        let post = rendered(LayerReg::Post, 0x0002_73b0);
        let text = post.as_str();
        assert!(text.contains("output_shift: -3"), "{text}");
        assert!(text.contains("bias_en: true"), "{text}");

        let lctl = rendered(LayerReg::Lctl, 0x0000_eb20);
        assert!(lctl.as_str().contains("relu: true"), "{}", lctl.as_str());
    }
}
