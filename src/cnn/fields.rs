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
