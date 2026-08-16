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

/// Declares a getter and builder for a multi-bit field at `[lsb + width - 1:lsb]`.
macro_rules! field {
    ($(#[$attr:meta])* $get:ident, $with:ident, $lsb:expr, $width:expr) => {
        $(#[$attr])*
        #[inline]
        pub const fn $get(self) -> u32 {
            (self.0 >> $lsb) & (u32::MAX >> (32 - $width))
        }

        $(#[$attr])*
        #[inline]
        pub const fn $with(self, value: u32) -> Self {
            let mask = u32::MAX >> (32 - $width);
            Self((self.0 & !(mask << $lsb)) | ((value & mask) << $lsb))
        }
    };
}

/// Declares a getter and builder for a single-bit field.
macro_rules! flag {
    ($(#[$attr:meta])* $get:ident, $with:ident, $bit:expr) => {
        $(#[$attr])*
        #[inline]
        pub const fn $get(self) -> bool {
            self.0 & (1 << $bit) != 0
        }

        $(#[$attr])*
        #[inline]
        pub const fn $with(self, value: bool) -> Self {
            Self((self.0 & !(1 << $bit)) | ((value as u32) << $bit))
        }
    };
}

/// Next layer
register! {
    Nxtlyr
}

impl Nxtlyr {
    field!(next, with_next, 0, 7);
    flag!(link_en, with_link_en, 7);
    flag!(stop, with_stop, 8);
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

/// Row count
register! {
    Rcnt
}

/// Column count
register! {
    Ccnt
}

macro_rules! count_register {
    ($name:ident) => {
        impl $name {
            field!(cnt, with_cnt, 0, 11);
            field!(pad_cnt, with_pad_cnt, 13, 2);
            flag!(pad_ena, with_pad_ena, 15);
            field!(diff, with_diff, 16, 16);
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

/// Pooling rows
register! {
    Prcnt
}

/// Pooling columns
register! {
    Pccnt
}

macro_rules! pool_register {
    ($name:ident) => {
        impl $name {
            field!(pool_cnt, with_pool_cnt, 0, 4);
            field!(pool_inc, with_pool_inc, 4, 4);
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

/// Pooling and multi-pass stride
register! {
    Stride
}

impl Stride {
    field!(stride, with_stride, 0, 4);
    field!(mp_stride, with_mp_stride, 4, 28);
}

impl core::fmt::Debug for Stride {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Stride")
            .field("stride", &self.stride())
            .field("mp_stride", &self.mp_stride())
            .finish()
    }
}

/// Data SRAM write pointer
register! {
    WptrBase
}

impl WptrBase {
    field!(offset, with_offset, 0, 13);
    field!(instance, with_instance, 13, 2);
    field!(group, with_group, 15, 6);
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

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({:#x})", stringify!($name), self.0)
            }
        }
    };
}

/// Write pointer time slot offset
value_register! {
    WptrToffs
}

/// Write pointer mask offset
value_register! {
    WptrMoffs
}

/// Write pointer multi-pass channel offset
value_register! {
    WptrChoffs
}

/// Data SRAM read pinter
value_register! {
    RptrBase
}

/// Layer control
register! {
    Lctl
}

impl Lctl {
    flag!(unnamed5, with_unnamed5, 5);
    flag!(chw, with_chw, 6);
    flag!(pool_ena, with_pool_ena, 7);
    flag!(maxpool, with_maxpool, 8);
    flag!(relu, with_relu, 9);
    flag!(global_wptr, with_global_wptr, 11);
    field!(siena, with_siena, 12, 4);
    flag!(wide_out, with_wide_out, 16);
    flag!(rd_ahead, with_rd_ahead, 17);
    field!(cprime_max, with_cprime_max, 18, 4);
    field!(rprime_max, with_rprime_max, 22, 4);
    field!(shift_cnt, with_shift_cnt, 26, 4);
    flag!(dw_bcast, with_dw_bcast, 29);
    flag!(bypass, with_bypass, 30);
}

impl core::fmt::Debug for Lctl {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("Lctl");
        s.field("chw", &self.chw())
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

/// Layer control 2
register! {
    Lctl2
}

impl Lctl2 {
    field!(maxpass, with_maxpass, 0, 4);
    field!(wptr_inc, with_wptr_inc, 4, 8);
    field!(xpch_max, with_xpch_max, 12, 9);
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

/// 1D convolution & element-wise configuration
register! {
    Oned
}

impl Oned {
    field!(tscnt_max, with_tscnt_max, 0, 4);
    field!(oned_sad, with_oned_sad, 4, 4);
    field!(oned_width, with_oned_width, 8, 4);
    flag!(oned_ena, with_oned_ena, 12);
    flag!(elt_ena, with_elt_ena, 13);
    field!(elt_fn, with_elt_fn, 14, 2);
    flag!(pool_first, with_pool_first, 16);
    flag!(elt_conv, with_elt_conv, 17);
    field!(operands, with_operands, 18, 4);
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

/// Last mask memory word
/// Not written for passthrough layers
register! {
    Mcnt1
}

impl Mcnt1 {
    field!(mexp_max, with_mexp_max, 0, 19);
}

impl core::fmt::Debug for Mcnt1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mcnt1")
            .field("mexp_max", &self.mexp_max())
            .finish()
    }
}

/// First mask memory word
register! {
    Mcnt2
}

impl Mcnt2 {
    field!(mexp_sad, with_mexp_sad, 0, 19);
}

impl core::fmt::Debug for Mcnt2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mcnt2")
            .field("mexp_sad", &self.mexp_sad())
            .finish()
    }
}

/// Output channel count minus one
register! {
    Ochan
}

impl Ochan {
    field!(ochan, with_ochan, 0, 32);
}

impl core::fmt::Debug for Ochan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ochan")
            .field("channels", &(self.ochan() + 1))
            .finish()
    }
}

/// TRAM pointer
register! {
    Tptr
}

impl Tptr {
    field!(tptr_max, with_tptr_max, 0, 16);
    field!(tptr_sad, with_tptr_sad, 16, 16);
}

impl core::fmt::Debug for Tptr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tptr")
            .field("tptr_max", &self.tptr_max())
            .field("tptr_sad", &self.tptr_sad())
            .finish()
    }
}

/// Processor and mask enables
register! {
    Ena
}

impl Ena {
    field!(proc_ena, with_proc_ena, 0, 16);
    field!(mask_ena, with_mask_ena, 16, 16);
}

impl core::fmt::Debug for Ena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ena")
            .field("proc_ena", &format_args!("{:#06x}", self.proc_ena()))
            .field("mask_ena", &format_args!("{:#06x}", self.mask_ena()))
            .finish()
    }
}

/// Post processing.
register! {
    Post
}

impl Post {
    field!(bias_addr, with_bias_addr, 0, 12);
    flag!(bias_en, with_bias_en, 12);
    field!(scale_mag, with_scale_mag, 13, 4);
    flag!(scale_dir, with_scale_dir, 17);
    field!(xpmp_cnt, with_xpmp_cnt, 18, 4);
    field!(wscale, with_wscale, 22, 2);
    flag!(ts_ena, with_ts_ena, 24);
    flag!(onexone_ena, with_onexone_ena, 25);
    flag!(act_abs, with_act_abs, 26);
    flag!(flatten_ena, with_flatten_ena, 27);
    flag!(xpose_ena, with_xpose_ena, 28);
    flag!(calcx4, with_calcx4, 29);
    flag!(dw_ena, with_dw_ena, 30);
    flag!(tcalc, with_tcalc, 31);

    #[inline]
    pub const fn shift_dir(self) -> ShiftDir {
        if self.scale_dir() {
            ShiftDir::Right
        } else {
            ShiftDir::Left
        }
    }

    #[inline]
    pub const fn with_shift_dir(self, dir: ShiftDir) -> Self {
        self.with_scale_dir(matches!(dir, ShiftDir::Right))
    }

    /// Weight-width compensation.
    #[inline]
    pub const fn weight_scale(self) -> WeightScale {
        WeightScale::from_code(self.wscale())
    }

    #[inline]
    pub const fn with_weight_scale(self, scale: WeightScale) -> Self {
        self.with_wscale(scale as u32)
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
    #[inline]
    pub const fn with_output_shift(self, shift: i32) -> Self {
        if shift < 0 {
            self.with_scale_mag((-shift) as u32).with_scale_dir(true)
        } else {
            self.with_scale_mag(shift as u32).with_scale_dir(false)
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

    #[inline]
    pub const fn with_pool_mode(self, mode: PoolMode) -> Self {
        self.with_maxpool(matches!(mode, PoolMode::Max))
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

    #[inline]
    pub const fn with_eltwise_fn(self, f: EltwiseFn) -> Self {
        self.with_elt_fn(f as u32)
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

    /// Write the activation into both registers, clearing the other bit.
    #[inline]
    pub const fn apply(self, lctl: Lctl, post: Post) -> (Lctl, Post) {
        (
            lctl.with_relu(matches!(self, Self::Relu)),
            post.with_act_abs(matches!(self, Self::Abs)),
        )
    }
}

/// its beautiful 🥹
pub trait LayerRegister: Copy + core::fmt::Debug {
    const REG: LayerReg;

    fn from_bits(bits: u32) -> Self;
    fn bits(self) -> u32;
}

macro_rules! layer_registers {
    ($($ty:ident => $reg:ident),* $(,)?) => {
        $(
            impl LayerRegister for $ty {
                const REG: LayerReg = LayerReg::$reg;

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
    const ROWS: [u32; 12] = [
        0x0001_0000,
        0x0001_806f,
        0x0001_807f,
        0x0001_80ff,
        0x0002_007f,
        0x0002_803f,
        0x0052_803e,
        0x0072_806e,
        0x00a2_807e,
        0x0142_80fe,
        0x0c40_0000,
        0x1000_0000,
    ];

    const COLUMNS: [u32; 10] = [
        0x0001_0000,
        0x0001_803b,
        0x0001_806f,
        0x0001_813f,
        0x0002_809e,
        0x0002_813e,
        0x0003_8004,
        0x0003_800c,
        0x000e_0000,
        0x0010_0000,
    ];

    const POOL: [u32; 5] = [0x1, 0x3, 0x6, 0xd, 0xf];

    const STRIDES: [u32; 8] = [
        0x0000_0010,
        0x0000_0020,
        0x0000_0021,
        0x0000_00c0,
        0x0000_0101,
        0x0000_03c3,
        0x0000_0e0d,
        0x0000_100f,
    ];

    const WPTR_BASES: [u32; 8] = [
        0x0000_0001,
        0x0000_0800,
        0x0000_2800,
        0x0000_42b9,
        0x0003_9000,
        0x0004_1fff,
        0x0006_0c80,
        0x0006_3db8,
    ];

    fn roundtrip_rcnt(bits: u32) {
        let r = Rcnt::from_bits(bits);
        let rebuilt = Rcnt::new()
            .with_cnt(r.cnt())
            .with_pad_cnt(r.pad_cnt())
            .with_pad_ena(r.pad_ena())
            .with_diff(r.diff());
        assert_eq!(rebuilt.bits(), bits, "Rcnt {bits:#010x}");
    }

    fn roundtrip_ccnt(bits: u32) {
        let r = Ccnt::from_bits(bits);
        let rebuilt = Ccnt::new()
            .with_cnt(r.cnt())
            .with_pad_cnt(r.pad_cnt())
            .with_pad_ena(r.pad_ena())
            .with_diff(r.diff());
        assert_eq!(rebuilt.bits(), bits, "Ccnt {bits:#010x}");
    }

    #[test]
    fn counts_roundtrip() {
        for bits in ROWS {
            roundtrip_rcnt(bits);
        }
        for bits in COLUMNS {
            roundtrip_ccnt(bits);
        }
    }

    #[test]
    fn pools_roundtrip() {
        for bits in POOL {
            let p = Prcnt::from_bits(bits);
            let rebuilt = Prcnt::new()
                .with_pool_cnt(p.pool_cnt())
                .with_pool_inc(p.pool_inc());
            assert_eq!(rebuilt.bits(), bits, "Prcnt {bits:#010x}");
        }
    }

    #[test]
    fn strides_roundtrip() {
        for bits in STRIDES {
            let s = Stride::from_bits(bits);
            let rebuilt = Stride::new()
                .with_stride(s.stride())
                .with_mp_stride(s.mp_stride());
            assert_eq!(rebuilt.bits(), bits, "Stride {bits:#010x}");
        }
    }

    #[test]
    fn write_pointers_roundtrip() {
        for bits in WPTR_BASES {
            let w = WptrBase::from_bits(bits);
            let rebuilt = WptrBase::new()
                .with_offset(w.offset())
                .with_instance(w.instance())
                .with_group(w.group());
            assert_eq!(rebuilt.bits(), bits, "WptrBase {bits:#010x}");
        }
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

        assert_eq!(WptrMoffs::from_bits(0x8000).bits(), 0x8000);
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

        assert_eq!(RptrBase::from_bits(0x2000).bits(), 0x2000);
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
        let linked = Nxtlyr::new().with_link_en(true).with_next(7);
        assert_eq!(linked.bits(), 0x87);
        assert_eq!(linked.next(), 7);

        // Link-layer mode on the final layer.
        let stopped = Nxtlyr::new().with_stop(true);
        assert_eq!(stopped.bits(), 0x100);

        // The snoop-loop override force-writes a link back to layer 0.
        let loop_back = Nxtlyr::from_bits(0x80);
        assert!(loop_back.link_en());
        assert_eq!(loop_back.next(), 0);
    }

    #[test]
    fn builders_overwrite_rather_than_or() {
        let r = Rcnt::new().with_cnt(0x7ff).with_cnt(1);
        assert_eq!(r.cnt(), 1);
        assert_eq!(r.bits(), 1);

        let n = Nxtlyr::new().with_stop(true).with_stop(false);
        assert_eq!(n.bits(), 0);
    }

    #[test]
    fn builders_mask_out_of_range_values() {
        assert_eq!(Nxtlyr::new().with_next(0xff).bits(), 0x7f);
        assert_eq!(Rcnt::new().with_pad_cnt(0xff).bits(), 0x6000);
        assert_eq!(Prcnt::new().with_pool_cnt(0xff).bits(), 0x0f);
    }

    const LCTLS: [u32; 12] = [
        0x0000_0920,
        0x0000_08a0,
        0x0000_0b20,
        0x0000_eb20,
        0x0001_e8a0,
        0x0080_6aa0,
        0x0080_cb20,
        0x0088_aba0,
        0x0089_e920,
        0x0100_ab20,
        0x0201_2920,
        0x2088_0b20,
    ];

    const LCTL2S: [u32; 9] = [
        0x0000_0001,
        0x0000_000f,
        0x0002_0000,
        0x000a_0002,
        0x0017_8035,
        0x0019_8011,
        0x001d_80bb,
        0x001f_80e3,
        0x001f_80ff,
    ];

    const ONEDS: [u32; 9] = [
        0x0000_0003,
        0x0000_0100,
        0x0000_0103,
        0x0000_1100,
        0x0000_1300,
        0x0000_1520,
        0x0000_1900,
        0x0004_6003,
        0x0005_6003,
    ];

    const MCNT1S: [u32; 6] = [
        0x0000_0008,
        0x0000_0078,
        0x0000_1210,
        0x0002_dbf8,
        0x0003_0d18,
        0x0003_9df8,
    ];

    const MCNT2S: [u32; 5] = [
        0x0000_0008,
        0x0000_0078,
        0x0000_1200,
        0x0001_9e00,
        0x0002_dc60,
    ];

    const OCHANS: [u32; 6] = [
        0x0000_0004,
        0x0000_000b,
        0x0000_00cf,
        0x0000_03ff,
        0x0000_0b3f,
        0x0000_3fff,
    ];

    const TPTRS: [u32; 6] = [
        0x0000_0002,
        0x0000_000f,
        0x0000_006f,
        0x0070_00a7,
        0x0140_027f,
        0x0460_04af,
    ];

    const ENAS: [u32; 11] = [
        0x0000_000f,
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
    ];

    /// Rebuilt through `shift_cnt`, which spans bit 29 and so subsumes
    /// `dw_bcast`.
    #[test]
    fn lctl_roundtrips() {
        for bits in LCTLS {
            let l = Lctl::from_bits(bits);
            let rebuilt = Lctl::new()
                .with_unnamed5(l.unnamed5())
                .with_chw(l.chw())
                .with_pool_ena(l.pool_ena())
                .with_maxpool(l.maxpool())
                .with_relu(l.relu())
                .with_global_wptr(l.global_wptr())
                .with_siena(l.siena())
                .with_wide_out(l.wide_out())
                .with_rd_ahead(l.rd_ahead())
                .with_cprime_max(l.cprime_max())
                .with_rprime_max(l.rprime_max())
                .with_shift_cnt(l.shift_cnt())
                .with_bypass(l.bypass());
            assert_eq!(rebuilt.bits(), bits, "Lctl {bits:#010x}");
        }
    }

    #[test]
    fn lctl2_roundtrips() {
        for bits in LCTL2S {
            let l = Lctl2::from_bits(bits);
            let rebuilt = Lctl2::new()
                .with_maxpass(l.maxpass())
                .with_wptr_inc(l.wptr_inc())
                .with_xpch_max(l.xpch_max());
            assert_eq!(rebuilt.bits(), bits, "Lctl2 {bits:#010x}");
        }
    }

    #[test]
    fn oned_roundtrips() {
        for bits in ONEDS {
            let o = Oned::from_bits(bits);
            let rebuilt = Oned::new()
                .with_tscnt_max(o.tscnt_max())
                .with_oned_sad(o.oned_sad())
                .with_oned_width(o.oned_width())
                .with_oned_ena(o.oned_ena())
                .with_elt_ena(o.elt_ena())
                .with_elt_fn(o.elt_fn())
                .with_pool_first(o.pool_first())
                .with_elt_conv(o.elt_conv())
                .with_operands(o.operands());
            assert_eq!(rebuilt.bits(), bits, "Oned {bits:#010x}");
        }
    }

    #[test]
    fn mask_counts_roundtrip() {
        for bits in MCNT1S {
            let m = Mcnt1::from_bits(bits);
            assert_eq!(Mcnt1::new().with_mexp_max(m.mexp_max()).bits(), bits);
        }
        for bits in MCNT2S {
            let m = Mcnt2::from_bits(bits);
            assert_eq!(Mcnt2::new().with_mexp_sad(m.mexp_sad()).bits(), bits);
        }
    }

    #[test]
    fn ochan_roundtrips_without_truncating() {
        for bits in OCHANS {
            let o = Ochan::from_bits(bits);
            assert_eq!(Ochan::new().with_ochan(o.ochan()).bits(), bits);
        }
        // imagenet layer 33 needs 14 bits; a [11:0] field would lose them.
        assert_eq!(Ochan::from_bits(0x3fff).ochan(), 0x3fff);
    }

    #[test]
    fn tptr_roundtrips() {
        for bits in TPTRS {
            let t = Tptr::from_bits(bits);
            let rebuilt = Tptr::new()
                .with_tptr_max(t.tptr_max())
                .with_tptr_sad(t.tptr_sad());
            assert_eq!(rebuilt.bits(), bits, "Tptr {bits:#010x}");
        }
    }

    #[test]
    fn ena_roundtrips() {
        for bits in ENAS {
            let e = Ena::from_bits(bits);
            let rebuilt = Ena::new()
                .with_proc_ena(e.proc_ena())
                .with_mask_ena(e.mask_ena());
            assert_eq!(rebuilt.bits(), bits, "Ena {bits:#010x}");
        }
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
        assert_eq!(slave.bits(), master.with_siena(0).bits());

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
        let l = Lctl::new().with_rd_ahead(true).with_shift_cnt(8);
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
        assert_eq!(pooled.bits(), o.with_pool_first(true).bits());
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

    #[test]
    fn post_roundtrips() {
        for bits in POSTS {
            let p = Post::from_bits(bits);
            let rebuilt = Post::new()
                .with_bias_addr(p.bias_addr())
                .with_bias_en(p.bias_en())
                .with_scale_mag(p.scale_mag())
                .with_scale_dir(p.scale_dir())
                .with_xpmp_cnt(p.xpmp_cnt())
                .with_wscale(p.wscale())
                .with_ts_ena(p.ts_ena())
                .with_onexone_ena(p.onexone_ena())
                .with_act_abs(p.act_abs())
                .with_flatten_ena(p.flatten_ena())
                .with_xpose_ena(p.xpose_ena())
                .with_calcx4(p.calcx4())
                .with_dw_ena(p.dw_ena())
                .with_tcalc(p.tcalc());
            assert_eq!(rebuilt.bits(), bits, "Post {bits:#010x}");
        }
    }

    /// The field the plan singles out as most error-prone: five bits of signed
    /// magnitude, not two's complement.
    #[test]
    fn output_shift_is_signed_magnitude() {
        // The worked example from the specification.
        let p = Post::new().with_output_shift(-3);
        assert_eq!(p.scale_mag(), 3);
        assert!(p.scale_dir());
        assert_eq!(p.bits() >> 13, 0b1_0011);
        assert_ne!(p.bits() >> 13, 0b1_1101, "encoded as two's complement");
        assert_eq!(p.output_shift(), -3);

        for shift in -15..=15 {
            let p = Post::new().with_output_shift(shift);
            assert_eq!(p.output_shift(), shift, "shift {shift}");
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
        assert_eq!(Post::new().with_output_shift(2).shift_dir(), ShiftDir::Left);
        assert_eq!(
            Post::new().with_output_shift(-2).shift_dir(),
            ShiftDir::Right
        );
        assert!(Post::new().with_shift_dir(ShiftDir::Right).scale_dir());
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
            assert_eq!(Post::new().with_weight_scale(s).weight_scale(), s);
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
            assert_eq!(Oned::new().with_eltwise_fn(f).eltwise_fn(), f);
        }
    }

    #[test]
    fn pool_mode_follows_maxpool() {
        assert_eq!(Lctl::from_bits(0x920).pool_mode(), PoolMode::Max);
        assert_eq!(Lctl::new().pool_mode(), PoolMode::Avg);
        assert!(Lctl::new().with_pool_mode(PoolMode::Max).maxpool());
    }

    /// Activation spans two registers, so it round-trips through both.
    #[test]
    fn activation_spans_lctl_and_post() {
        let lctl = Lctl::from_bits(0x0000_eb20);
        let post = Post::from_bits(0x0000_2000);
        assert_eq!(Activation::decode(lctl, post), Some(Activation::Relu));

        for act in [Activation::None, Activation::Relu, Activation::Abs] {
            let (l, p) = act.apply(Lctl::new(), Post::new());
            assert_eq!(Activation::decode(l, p), Some(act));
        }

        // Abs lives in POST, not LCTL.
        let (l, p) = Activation::Abs.apply(Lctl::new(), Post::new());
        assert!(!l.relu());
        assert!(p.act_abs());

        // Both bits set is not a valid encoding.
        let both = Lctl::new().with_relu(true);
        let abs = Post::new().with_act_abs(true);
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
