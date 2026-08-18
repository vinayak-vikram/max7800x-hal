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
