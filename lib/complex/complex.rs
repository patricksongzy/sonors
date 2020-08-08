use std::ops;

#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(feature = "f32")]
pub use std::f32 as floats;
#[cfg(not(feature = "f32"))]
pub type Float = f64;
#[cfg(not(feature = "f32"))]
pub use std::f64 as floats;

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

impl ops::Mul<Complex> for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(self.re * other.re - self.im * other.im, self.re * other.im + self
.im * other.re)
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

