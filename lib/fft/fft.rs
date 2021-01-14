// Algorithms based off 'Real-Valued Fast Fourier Transform Algorithms' by Sorensen et al.
// Tests based off [](https://www.dsprelated.com/showthread/comp.dsp/71595-1.php),
// and 'Testing Multivariate Linear Functions: Overcoming the Generator Bottleneck' by Ergun.
//
// @article{Sorensen1987RealvaluedFF,
//   title={Real-valued fast Fourier transform algorithms},
//   author={H. Sorensen and D. Jones and M. Heideman and C. Burrus},
//   journal={IEEE Trans. Acoust. Speech Signal Process.},
//   year={1987},
//   volume={35},
//   pages={849-863}
// }
//
// @INPROCEEDINGS{Ergün94testingmultivariate,
//   author = {Funda Ergün},
//   title = {Testing Multivariate Linear Functions: Overcoming the Generator Bottleneck},
//   booktitle = {Proc. 27th STOC},
//   year = {1994},
//   pages = {407--416}
// }
//

use complex::complex::*;

pub fn get_frequencies(signal_length: usize, sample_period: Float) -> Vec<Float> {
    let scale = 1.0 as Float / (sample_period * signal_length as Float);
    (0..=(signal_length as i32 - 1) / 2).chain(-((signal_length / 2) as i32)..0).map(|value| value as Float * scale).collect()
}

pub fn get_real_frequencies(signal_length: usize, sample_period: Float) -> Vec<Float> {
    let scale = 1.0 as Float / (sample_period * signal_length as Float);
    (0..=signal_length / 2).map(|value| value as Float * scale).collect()
}

pub fn get_rotated_signal(signal: &mut Vec<Float>) {
    let signal_length = signal.len();
    signal.rotate_left((signal_length as Float / 2.0).ceil() as usize);
}

pub fn dft(input_signal: Vec<Complex>) -> Vec<Complex> {
    let signal_length = input_signal.len();
    let mut results = vec![Complex::new(0.0, 0.0); signal_length];
    for i in 0..signal_length {
        for j in 0..signal_length {
            let angle = 2.0 * std::f64::consts::PI * (i * j) as Float / signal_length as Float;
            results[i] += input_signal[j] * Complex::new(angle.cos(), -angle.sin());
        }
    }

    results
}

/// Computes a radix-2 Fast Fourier Transform. The output sequence is a `Vec` of the complex
/// results. This function writes to a new `Vec` instead of overwriting the original.
pub fn fft(input_signal: Vec<Complex>) -> Vec<Complex> {
    let signal_length: usize = input_signal.len();

    if signal_length == 1 {
        return input_signal;
    }

    // split into tuple of even, and odd indexed vectors
    let partitioned: (Vec<Complex>, Vec<Complex>) = partition_with_index(&input_signal, |&(i, _)| i % 2 == 0);

    // perform fft on even indices
    let mut output_signal: Vec<Complex> = fft(partitioned.0);
    // perform fft on odd indices
    output_signal.extend(fft(partitioned.1));

    // precompute coefficient
    let coefficient: Float = (2.0 * std::f64::consts::PI / signal_length as f64) as Float;
    for k in 0..signal_length / 2 {
        let t = output_signal[k];
        let angle = coefficient * k as Float;
        let rotating_vectors = Complex::new(angle.cos(), -angle.sin());

        // equivalent to exponentiation
        let product = rotating_vectors * output_signal[k + signal_length / 2];

        // leverage symmetry to only perform half calculations
        output_signal[k] = t + product;
        output_signal[k + signal_length / 2] = t - product;
    }

    output_signal
}

/// Computes a radix-2 Fast Fourier Transform. The output sequence is a `Vec` of the complex
/// results. This function writes overwrites the original `Vec`, instead of creating a new one.
pub fn iterative_fft(signal: &mut Vec<Complex>) {
    let signal_length: usize = signal.len();
    bit_reverse_copy(signal);

    let mut m_halves = 1;
    for _ in 1..=log2(signal_length) {
        let m = 2 * m_halves;

        let angle: Float = 2.0 * std::f64::consts::PI / m as Float;
        let wm = Complex::new(angle.cos(), -angle.sin());

        for k in (0..signal_length).step_by(m) { 
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..m_halves {
                let t = w * signal[k + j + m_halves];
                let u = signal[k + j];

                signal[k + j] = u + t;
                signal[k + j + m_halves] = u - t;

                w *= wm;
            }
        }

        m_halves = m;
    }
}

/// Calculates the logarithm, base 2 of a `usize`. This is made easy due to the base of 2.
fn log2(input: usize) -> usize {
    std::mem::size_of::<usize>() * 8 - input.leading_zeros() as usize - 1
}

/// Overwrites an input `Vec` in a bit-reversed order.
fn bit_reverse_copy<T>(input: &mut Vec<T>)
where
    T: Copy,
{
    let input_length = input.len();
    let leading_zeros = (input_length - 1).leading_zeros();
    for k in 0..input_length {
        let j = k.reverse_bits() >> leading_zeros;
        if k < j {
            let t = input[k];
            input[k] = input[j];
            input[j] = t;
        }
    }
}

/// Calculates two real-valued Fast Fourier Transforms similtaneously.
pub fn combined_rfft(left: Vec<Float>, right: Vec<Float>) -> (Vec<Complex>, Vec<Complex>) {
    let signal_length: usize = left.len();

    // store the left, and right signals in a `Vec<Complex>` of the same length
    let mut combined_signal: Vec<Complex> = vec![Complex::new(0.0, 0.0); signal_length];
    for ((combined, left), right) in combined_signal.iter_mut().zip(&left).zip(&right) {
        *combined = Complex::new(*left, *right);
    }

    let combined_output = fft(combined_signal);
    let mut output_left: Vec<Complex> = vec![Complex::new(0.0, 0.0); signal_length];
    let mut output_right: Vec<Complex> = vec![Complex::new(0.0, 0.0); signal_length];

    output_left[0] = Complex::new(combined_output[0].re, 0.0);
    output_right[0] = Complex::new(combined_output[0].im, 0.0);
    for k in 1..signal_length {
        output_left[k] = Complex::new(0.5, 0.0) * (combined_output[k] + combined_output[signal_length - k].conj());
        // the right signal was stored as the complex values
        output_right[k] = Complex::new(0.0, 0.5) * (combined_output[signal_length - k].conj() - combined_output[k]);
    }
    
    (output_left, output_right)
}

/// Computes a real-valued Fast Fourier Transform through a `combined_rfft` of the even, and odd
/// values.
pub fn rfft(input_signal: Vec<Float>) -> Vec<Complex> {
    let signal_length: usize = input_signal.len();

    let (evens, odds): (Vec<Float>, Vec<Float>) = partition_with_index(&input_signal, |&(i, _)| i % 2 == 0);

    let (evens, odds) = combined_rfft(evens, odds);
    let mut output_signal: Vec<Complex> = evens;
    output_signal.extend(odds);

    let t = output_signal[0] - output_signal[signal_length / 2];

    let coefficient: Float = (2.0 * std::f64::consts::PI / signal_length as f64) as Float;
    for k in 0..signal_length / 2 {
        let angle = coefficient * k as Float;
        let rotating_vectors = Complex::new(angle.cos(), -angle.sin());

        let product = rotating_vectors * output_signal[k + signal_length / 2];
        output_signal[k] += product;
    }

    output_signal[signal_length / 2] = t;

    for k in 1..signal_length / 2 {
        output_signal[signal_length - k] = output_signal[k].conj();
    }
    
    output_signal
}

pub fn create_rotating_vectors(signal_length: usize) -> Vec<Complex> {
    let coefficient: Float = (2.0 * std::f64::consts::PI) as Float;
    let mut m = 2;

    let log_length = log2(signal_length);
    let mut rotating_vectors = vec![Complex::new(0.0, 0.0); log_length - 1];

    for s in 2..=log_length {
        m *= 2;

        let angle = coefficient / m as Float;
        rotating_vectors[s - 2] = Complex::new(angle.cos(), -angle.sin());
    }

    rotating_vectors
}

pub fn iterative_rfft_once(signal: &mut Vec<Float>) {
    iterative_rfft(signal, &create_rotating_vectors(signal.len()));
}

pub fn create_hann(signal_length: usize) -> Vec<Float> {
    let coefficient = (2.0 * std::f64::consts::PI / (signal_length - 1) as f64) as Float;
    (0..signal_length).map(|i| 0.5 - 0.5 * (coefficient * i as Float).cos()).collect::<Vec<Float>>()
}

/// Computes the radix-2 real-valued Fast fourier Transform. The output sequence is in the format
/// `{Re[0], Re[1], ..., Re[N / 2], Im[N / 2 - 1], Im[N / 2 - 2], ..., Im[1]}`, where `Im[0]`, and
/// `Im[N / 2]` are both `0`.
/// This function is based off the provided radix-2 real-valued FFT butterfly diagram by Sorensen
/// et al.
pub fn iterative_rfft(signal: &mut Vec<Float>, rotating_vectors: &Vec<Complex>) {
    let signal_length: usize = signal.len();
    bit_reverse_copy(signal);

    // length two butterflies where twiddle is `1`
    for i in (0..signal_length).step_by(2) {
        let t = signal[i];
        signal[i] = t + signal[i + 1];
        signal[i + 1] = t - signal[i + 1];
    }

    let mut m_halves = 1;
    for s in 2..=log2(signal_length) {
        let m_quarters = m_halves;
        m_halves = 2 * m_quarters;
        let m = 2 * m_halves;
        
        let wm = rotating_vectors[s - 2];

        // split into `N / m` groups
        for k in (0..signal_length).step_by(m) {
            // initialising `w` as `wm` is needed since some butterflies are skipped, and doing so reduces necessary trigonometric calculations
            let mut w = wm;

            let t = signal[k];
            // twiddle is `1` for the first element of the group
            signal[k] = t + signal[k + m_halves];
            signal[k + m_halves] = t - signal[k + m_halves];

            // the `m / 4`, and `3 * m / 4` (complex conjugates) butterflies only require a sign change, since their twiddle is `-i`
            signal[k + 3 * m_quarters] = -signal[k + 3 * m_quarters];

            // compute output from `[0, m / 4]`, and `[m / 2, 3 * m / 4]`
            for j in 1..m_quarters {
                // calculate and store the real values in [0, m / 4], and [m / 2, 3 * m / 4]
                let i_re = (k + j, k + j + m_halves);
                // store the imaginary values at the `(m - i) mod m`th position, redundant due to symmetry
                let i_im = (m + k - j, m_halves + k - j);

                let t = w * Complex::new(signal[i_re.1], signal[i_im.0]);

                // must use value at `i_im.1` before it is overwritten
                signal[i_im.0] = signal[i_im.1] + t.im;
                signal[i_re.1] = -signal[i_im.1] + t.im;

                signal[i_im.1] = signal[i_re.0] - t.re;
                signal[i_re.0] = signal[i_re.0] + t.re;

                w *= wm;
            }
        }
    }
}

/// Performs a partition with a predicate which can return `true`, or `false` based on index, and
/// value.
/// Based off of the Rust partition code, which is 'released under both the MIT licence, and the
/// Apache Licence (Version 2.0), with portions covered by various BSD-like licences.'
/// For more information, [](https://github.com/rust-lang/rust).
fn partition_with_index<D, B, F>(data: D, f: F) -> (B, B)
where
    D: IntoIterator + Sized,
    B: Default + Extend<D::Item>,
    F: FnMut(&(usize, D::Item)) -> bool,
{
    #[inline]
    fn extend<'a, T, B: Extend<T>>(
        mut f: impl FnMut(&(usize, T)) -> bool + 'a,
        left: &'a mut B,
        right: &'a mut B,
    ) -> impl FnMut((usize, T)) + 'a {
        // `p` is the index-value pair
        move |p| {
            if f(&p) {
                left.extend(Some(p.1));
            } else {
                right.extend(Some(p.1));
            }
        }
    }

    let mut left: B = Default::default();
    let mut right: B = Default::default();

    data.into_iter().enumerate().for_each(extend(f, &mut left, &mut right));

    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::ops::*;
    use rand::Rng;
    use rand::distributions;

    // test vectors
    const SIG_ZEROS: [Float; 8] = [0.0; 8];
    const ZEROS_FFT: [Complex; 8] = [Complex::new(0.0, 0.0); 8];
    const SIG_ONES: [Float; 8] = [1.0; 8];
    // useful to verify that the first value is calculated properly
    const ONES_FFT: [Complex; 8] = {
        let mut ones_fft = [Complex::new(0.0, 0.0); 8];
        ones_fft[0] = Complex::new(8.0, 0.0);
        ones_fft
    };
    const SIG_ALTERNATING: [Float; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    // useful to verify that the value at `N / 2` is calculated properly
    const ALTERNATING_FFT: [Complex; 8] = {
        let mut alternating_fft = [Complex::new(0.0, 0.0); 8];
        alternating_fft[alternating_fft.len() / 2] = Complex::new(8.0, 0.0);
        alternating_fft
    };

    fn create_signal(frequency: Float, signal_length: usize) -> Vec<Float> {
        let coefficient: Float = (2.0 * std::f64::consts::PI) as Float * frequency;
        (0..signal_length).map(|x| (coefficient * x as Float).sin()).collect()
    }

    fn wrap_real(signal: &Vec<Float>) -> Vec<Complex> {
        signal.iter().map(|x| Complex::new(*x, 0.0)).collect()
    }

    #[test]
    fn test_frequencies() {
        let frequencies = get_frequencies(8, 0.5 as Float);
        let expected = vec![0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25];

        for (output, target) in frequencies.iter().zip(&expected) {
            assert_relative_eq!(output, target);
        }
    }

    #[test]
    fn test_real_frequencies() {
        let frequencies = get_real_frequencies(8, 0.5 as Float);
        let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        for (output, target) in frequencies.iter().zip(&expected) {
            assert_relative_eq!(output, target);
        }
    }

    #[test]
    fn test_fft_baseline() {
        let sig_sine = wrap_real(&create_signal(0.5, 128));
        
        for (output, target) in fft(sig_sine.clone()).iter().zip(&dft(sig_sine)) {
            assert_relative_eq!(output, target);
        }
    }

    #[test]
    fn test_rfft_baseline() {
        let sig_sine = create_signal(0.5, 128);
        let sine_dft = dft(wrap_real(&sig_sine));

        for (output, target) in rfft(sig_sine).iter().zip(&sine_dft) {
            assert_relative_eq!(output, target);
        }
    }

    #[test]
    fn test_iterative_rfft_baseline() {
        let sig_sine = create_signal(0.5, 128);
        let sine_dft = dft(wrap_real(&sig_sine));

        let mut signal = sig_sine.clone();
        let signal_length = signal.len();
        iterative_rfft_once(&mut signal);

        for (output, target) in (&signal[..=signal_length / 2]).iter().zip(&sine_dft[..=signal_length / 2]) {
            assert_relative_eq!(output, &target.re);
        }

        for (target, output) in (&sine_dft[1..signal_length / 2]).iter().rev().zip(&signal[signal_length / 2 + 1..]) {
            assert_relative_eq!(output, &target.im);
        }
    }

    #[test]
    fn test_rfft_zeros() {
        let output_signal = rfft(SIG_ZEROS.to_vec());
        assert_eq!(output_signal, ZEROS_FFT);
    }

    #[test]
    fn test_rfft_ones() {
        let output_signal = rfft(SIG_ONES.to_vec());
        assert_eq!(output_signal, ONES_FFT);
    }

    #[test]
    fn test_rfft_alternating() {
        let output_signal = rfft(SIG_ALTERNATING.to_vec());
        assert_eq!(output_signal, ALTERNATING_FFT);
    }

    #[test]
    fn test_combined_rfft_alternating() {
        let (left, right) = combined_rfft(SIG_ALTERNATING.to_vec(), SIG_ALTERNATING.to_vec());
        assert_eq!(left, ALTERNATING_FFT);
        assert_eq!(right, ALTERNATING_FFT);
    }

    #[test]
    fn test_iterative_rfft() {
        let mut signal = SIG_ALTERNATING.to_vec();
        iterative_rfft_once(&mut signal);
        assert_eq!(signal[..=signal.len() / 2], ALTERNATING_FFT[0..=ALTERNATING_FFT.len() / 2]);
        assert_eq!(signal[signal.len() / 2 + 1..], [0.0; ALTERNATING_FFT.len() / 2 - 1]);
    }

    #[test]
    fn test_fft_linearity() {
        test_linearity(&fft, 1e-8);
    }

    #[test]
    fn test_iterative_fft_linearity() {
        test_linearity(&|signal| {
            let mut output = signal.clone();
            iterative_fft(&mut output);
            output
        }, 1e-8);
    }

    #[test]
    fn test_rfft_linearity() {
        test_linearity(&rfft, 1e-8);
    }

    #[test]
    fn test_iterative_rfft_linearity() {
        test_linearity(&|signal| {
            let mut output = signal.clone();
            iterative_rfft_once(&mut output);
            output
        }, 1e-8);
    }

    fn test_linearity<T, B>(f: &dyn Fn(Vec<T>) -> Vec<B>, epsilon: B::Epsilon)
    where
        T: Add + Mul<Output=T> + Mul<B, Output=B> + AddAssign + MulAssign + Copy + Clone + Debug,
        B: Add + Mul<Output=B> + Mul<T, Output=B> + AddAssign + MulAssign + MulAssign<T> + approx::RelativeEq + Copy + Debug,
        B::Epsilon: Copy,
        distributions::Standard: distributions::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let coefficient_left: T = rng.gen::<T>();
        let coefficient_right: T = rng.gen::<T>();
        let lefts: Vec<T> = (0..8).map(|_| rng.gen::<T>()).collect();
        let rights: Vec<T> = (0..8).map(|_| rng.gen::<T>()).collect();
        let mut input_signal: Vec<T> = lefts.clone();
        {
            for (input, right) in input_signal.iter_mut().zip(&rights) {
                *input *= coefficient_left;
                *input += coefficient_right * *right;
            }
        }

        let output_signal = f(input_signal);
        let mut output_equiv = f(lefts);
        let outputs_right = f(rights);
        {
            for (equivalent, output_right) in output_equiv.iter_mut().zip(&outputs_right) {
                *equivalent *= coefficient_left;
                *equivalent += coefficient_right * *output_right;
            }
        }

        for (output, equivalent) in output_signal.iter().zip(&output_equiv) {
            assert_relative_eq!(output, equivalent, epsilon=epsilon);
        }
    }
}

