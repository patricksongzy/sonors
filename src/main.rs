use fft::fft::*;
use complex::complex::*;

fn main() {
    let mut signal = create_signal(0.5, 128);
    rfft_r2(&mut signal);

    println!("{:?}", &signal[..64]);
    println!("{:?}", &signal[65..]);
}

fn create_signal(frequency: Float, signal_length: usize) -> Vec<Float> {
    let coefficient: Float = 2.0 * std::f64::consts::PI * frequency;
    (0..signal_length).map(|x| (coefficient * x as Float).sin()).collect()
}
