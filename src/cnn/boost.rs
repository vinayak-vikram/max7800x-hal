//! Boost circuit for the accelerator supply
//!
//! A GPIO driving an external load switch; which pin is a board decision.

use embedded_hal::digital::OutputPin;

/// Which level turns the load switch on
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BoostPolarity {
    #[default]
    ActiveHigh,
    ActiveLow,
}

/// The GPIO driving the boost load switch
pub struct CnnBoost<P: OutputPin> {
    pin: P,
    polarity: BoostPolarity,
}

impl<P: OutputPin> CnnBoost<P> {
    /// Take a pin that drives the switch on when high
    pub fn new(pin: P) -> Self {
        Self {
            pin,
            polarity: BoostPolarity::ActiveHigh,
        }
    }

    /// Take a pin whose active level is `polarity`
    pub fn with_polarity(pin: P, polarity: BoostPolarity) -> Self {
        Self { pin, polarity }
    }

    /// Raise the supply, before running the accelerator at speed
    pub fn enable(&mut self) -> Result<(), P::Error> {
        self.drive(true)
    }

    /// Drop back to the unboosted supply
    pub fn disable(&mut self) -> Result<(), P::Error> {
        self.drive(false)
    }

    pub const fn polarity(&self) -> BoostPolarity {
        self.polarity
    }

    /// Give the pin back, leaving it wherever it was last driven
    pub fn release(self) -> P {
        self.pin
    }

    fn drive(&mut self, on: bool) -> Result<(), P::Error> {
        if on == matches!(self.polarity, BoostPolarity::ActiveHigh) {
            self.pin.set_high()
        } else {
            self.pin.set_low()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_hal::digital::ErrorType;

    #[derive(Default)]
    struct FakePin {
        high: bool,
    }

    impl ErrorType for FakePin {
        type Error = core::convert::Infallible;
    }

    impl OutputPin for FakePin {
        fn set_high(&mut self) -> Result<(), Self::Error> {
            self.high = true;
            Ok(())
        }

        fn set_low(&mut self) -> Result<(), Self::Error> {
            self.high = false;
            Ok(())
        }
    }

    /// The C helper drives the pin high to enable, which is the default here.
    #[test]
    fn active_high_drives_high_to_enable() {
        let mut boost = CnnBoost::new(FakePin::default());
        boost.enable().unwrap();
        assert!(boost.release().high);

        let mut boost = CnnBoost::new(FakePin::default());
        boost.disable().unwrap();
        assert!(!boost.release().high);
    }

    /// A load switch with an inverting enable is a board decision, not a
    /// property of the accelerator, so both polarities have to work.
    #[test]
    fn active_low_inverts_both_directions() {
        let mut boost = CnnBoost::with_polarity(FakePin::default(), BoostPolarity::ActiveLow);
        boost.enable().unwrap();
        assert!(!boost.release().high);

        let mut boost = CnnBoost::with_polarity(FakePin::default(), BoostPolarity::ActiveLow);
        boost.disable().unwrap();
        assert!(boost.release().high);
    }
}
