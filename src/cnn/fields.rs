//! # CNN Layer Register Values
//!

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
}
