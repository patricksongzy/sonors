// Algorithms based off 'Real-Valued Fast Fourier Transform Algorithms' by Sorensen et al.  Tests based off [](https://www.dsprelated.com/showthread/comp.dsp/71595-1.php), and 'Testing Multivariate Linear Functions: Overcoming the Generator Bottleneck' by Ergun.

use complex::complex::*;

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
    let coefficient: Float = 2.0 * std::f64::consts::PI / signal_length as Float;
    for k in 0..signal_length / 2 {
        let t = output_signal[k];
        let angle = coefficient * k as Float;
        let rotating_vector = Complex::new(angle.cos(), -angle.sin());

        // equivalent to exponentiation
        let product = rotating_vector * output_signal[k + signal_length / 2];

        // leverage symmetry to only perform half calculations
        output_signal[k] = t + product;
        output_signal[k + signal_length / 2] = t - product;
    }

    output_signal
}

pub fn iterative_fft(mut signal: Vec<Complex>) {
    let signal_length: usize = signal.len();
    bit_reverse_copy(&mut signal);

    let mut m_halves = 1;
    for _ in 1..=log2(signal_length) {
        let m = 2 * m_halves;
        m_halves = m;

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
    }
}

fn log2(input: usize) -> usize {
    std::mem::size_of::<usize>() * 8 - input.leading_zeros() as usize - 1
}

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
        // result[k.reverse_bits() >> leading_zeros] = input[k].clone();
    }
}

pub fn combined_rfft(left: Vec<Float>, right: Vec<Float>) -> (Vec<Complex>, Vec<Complex>) {
    let signal_length: usize = left.len();

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
        output_right[k] = Complex::new(0.0, 0.5) * (combined_output[signal_length - k].conj() - combined_output[k]);
    }
    
    (output_left, output_right)
}

pub fn rfft(input_signal: Vec<Float>) -> Vec<Complex> {
    let signal_length: usize = input_signal.len();

    let (evens, odds): (Vec<Float>, Vec<Float>) = partition_with_index(&input_signal, |&(i, _)| i % 2 == 0);

    let (evens, odds) = combined_rfft(evens, odds);
    let mut output_signal: Vec<Complex> = evens;
    output_signal.extend(odds);

    let t = output_signal[0] - output_signal[signal_length / 2];

    let coefficient: Float = 2.0 * std::f64::consts::PI / signal_length as Float;
    for k in 0..signal_length / 2 {
        let angle = coefficient * k as Float;
        let rotating_vector = Complex::new(angle.cos(), -angle.sin());

        let product = rotating_vector * output_signal[k + signal_length / 2];
        output_signal[k] += product;
    }

    output_signal[signal_length / 2] = t;

    for k in 1..signal_length / 2 {
        output_signal[signal_length - k] = output_signal[k].conj();
    }
    
    output_signal
}

/// Computes the radix-2 real-valued Fast fourier Transform. The output sequence is in the format
/// `{Re[0], Re[1], ..., Re[N / 2], Im[N / 2 - 1], Im[N / 2 - 2], ..., Im[1]}`, where `Im[0]`, and
/// `Im[N / 2]` are both `0`.
/// This function is based off the provided radix-2 real-valued FFT butterfly diagram by Sorensen
/// et al.
pub fn rfft_r2(signal: &mut Vec<Float>) {
    let signal_length: usize = signal.len();
    bit_reverse_copy(signal);

    // length two butterflies where twiddle is `1`
    for i in (0..signal_length).step_by(2) {
        let t = signal[i];
        signal[i] = t + signal[i + 1];
        signal[i + 1] = t - signal[i + 1];
    }

    let mut m_halves = 1;
    for _ in 2..=log2(signal_length) {
        let m_quarters = m_halves;
        m_halves = 2 * m_quarters;
        let m = 2 * m_halves;

        let angle: Float = 2.0 * std::f64::consts::PI / m as Float;
        let wm = Complex::new(angle.cos(), -angle.sin());

        // split into `N / m` groups
        for k in (0..signal_length).step_by(m) {
            // initialising `w` as `wm` is needed since some butterflies are skipped, and doing so reduces necessary trigonometric calculations
            let mut w = wm;

            let t = signal[k];
            // twiddle is `1` for the first element of the group
            signal[k] = t + signal[k + m_halves];
            signal[k + m_halves] = t - signal[k + m_halves];

            // the `N / 4`, and `3 * N / 4` (complex conjugates) butterflies only require a sign change, since their twiddle is `-i`
            signal[k + 3 * m_quarters] = -signal[k + 3 * m_quarters];

            // compute output from `[0, N / 4]`, and `[N / 2, 3 * N / 4]`
            for j in 1..m_quarters {
                // calculate and store the real values in [0, N / 4], and [n / 2, 3 * N / 4]
                let i_re = (k + j, k + j + m_halves);
                // store the imaginary values at the `(N - i) mod N`th position, redundant due to symmetry
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
    use rand::Rng;

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

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() > $d {
                panic!();
            }
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
    fn test_rfft_r2() {
        let mut signal = SIG_ALTERNATING.to_vec();
        rfft_r2(&mut signal);
        println!("{:?}", signal);
    }

    #[test]
    fn test_rfft_linearity() {
        let mut rng = rand::thread_rng();
        let coefficient_left: Float = rng.gen::<Float>();
        let coefficient_right: Float = rng.gen::<Float>();
        let lefts: Vec<Float> = (0..8).map(|_| rng.gen::<Float>()).collect();
        let rights: Vec<Float> = (0..8).map(|_| rng.gen::<Float>()).collect();
        let mut input_signal: Vec<Float> = lefts.clone();
        {
            for (input, right) in input_signal.iter_mut().zip(&rights) {
                *input *= coefficient_left;
                *input += coefficient_right * right;
            }
        }

        let output_signal = rfft(input_signal);
        let mut output_equiv = rfft(lefts);
        let outputs_right = rfft(rights);
        {
            for (equivalent, output_right) in output_equiv.iter_mut().zip(&outputs_right) {
                *equivalent *= Complex::new(coefficient_left, 0.0);
                *equivalent += Complex::new(coefficient_right, 0.0) * *output_right;
            }
        }

        for (output, equivalent) in output_signal.iter().zip(output_equiv) {
            assert_delta!(output.re, equivalent.re, 1e-8);
            assert_delta!(output.im, equivalent.im, 1e-8);
        }
    }
}

