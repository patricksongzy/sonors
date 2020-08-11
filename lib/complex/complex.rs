use std::ops;
use rand::Rng;
use rand::distributions::{Standard, Distribution};

use approx::*;

#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
pub type Float = f64;

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Complex {
    pub re: Float,
    pub im: Float,
}

impl Complex {
    #[inline]
    pub const fn new(re: Float, im: Float) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }
}

impl Distribution<Complex> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex {
        let (re, im) = rng.gen();
        Complex::new(re, im)
    }
}

impl PartialEq<Complex> for Float {
    fn eq(&self, other: &Complex) -> bool {
        *self == other.re && other.im == 0.0
    }
}

impl PartialEq<Float> for Complex {
    fn eq(&self, other: &Float) -> bool {
        self.re == *other && self.im == 0.0
    }
}

impl AbsDiffEq for Complex {
    type Epsilon = <Float as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> <Float as AbsDiffEq>::Epsilon {
        Float::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: <Float as AbsDiffEq>::Epsilon) -> bool {
        Float::abs_diff_eq(&self.re, &other.re, epsilon) && Float::abs_diff_eq(&self.im, &other.im, epsilon)
    }
}

impl RelativeEq for Complex {
    fn default_max_relative() -> <Float as AbsDiffEq>::Epsilon {
        Float::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: <Float as AbsDiffEq>::Epsilon, max_relative: <Float as AbsDiffEq>::Epsilon) -> bool {
        Float::relative_eq(&self.re, &other.re, epsilon, max_relative) && Float::relative_eq(&self.im, &other.im, epsilon, max_relative)
    }
}

impl ops::Mul<Complex> for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(self.re * other.re - self.im * other.im, self.re * other.im + self
.im * other.re)
    }
}

impl ops::Mul<Float> for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, other: Float) -> Self {
        Self::new(self.re * other, self.im * other)
    }
}

impl ops::Mul<Complex> for Float {
    type Output = Complex;

    #[inline]
    fn mul(self, other: Complex) -> Complex {
        Complex::new(self * other.re, self * other.im)
    }
}

impl ops::MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        let a = self.re.clone();
        self.re *= other.re;
        self.re -= self.im * other.im;

        self.im *= other.re;
        self.im += a * other.im;
    }
}

impl ops::MulAssign<Float> for Complex {
    #[inline]
    fn mul_assign(&mut self, other: Float) {
        self.re *= other;
        self.im *= other;
    }
}

impl ops::Add<Complex> for Complex {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl ops::AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl ops::Sub<Complex> for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl ops::SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl std::fmt::Debug for Complex {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut sign = "+";
        if self.im.is_sign_negative() {
            sign = "-";
        }

        write!(f, "{} {} {}i", self.re, sign, self.im.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplication() {
        let product = Complex::new(-5.0, 3.0) * Complex::new(7.0, 8.0);
        assert_eq!(product, Complex::new(-59.0, -19.0));
    }

    #[test]
    fn test_signs() {
        let product = Complex::new(0.0, -1.0) * Complex::new(0.0, 1.0);
        assert_eq!(product, Complex::new(1.0, 0.0));
    }
}

