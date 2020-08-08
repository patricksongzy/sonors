use fft::fft::*;
use complex::complex::*;

fn main() {
    let input_signal = create_signal(0.5, 128);
    let output_signal = rfft(input_signal);

    println!("{:?}", output_signal);
}

fn create_signal(frequency: Float, signal_length: usize) -> Vec<Float> {
    let coefficient: Float = 2.0 * std::f64::consts::PI * frequency;
    (0..signal_length).map(|x| (coefficient * x as Float).sin()).collect()
}
