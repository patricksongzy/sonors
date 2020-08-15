use portaudio as pa;

use std::sync::mpsc::*;
use std::cmp;

use fft::fft::*;
use complex::complex::Float;

const WINDOW_SIZE: usize = 256;
const N_OVERLAP: usize = 128;

fn circular_read<T>(circular_buffer: &Vec<T>, read_ptr: usize, read_len: usize) -> Vec<T>
where
    T: Copy,
{
    let buffer_length = circular_buffer.len();
    let offset: i32 = (read_ptr + read_len) as i32 - buffer_length as i32;
    let end_ptr = cmp::min(read_ptr + read_len, buffer_length);

    let mut result = circular_buffer[read_ptr..end_ptr].to_vec();

    if offset > 0 {
        result.extend(circular_buffer[0..offset as usize].to_vec());
    }

    result
}

fn run() -> Result<(), pa::Error> {
    let pa = pa::PortAudio::new()?;
    let mic_index = pa.default_input_device()?;
    let mic = pa.device_info(mic_index)?;

    let input_params = pa::StreamParameters::<f32>::new(mic_index, 1, true, mic.default_low_input_latency);

    let input_settings = pa::InputStreamSettings::new(input_params, mic.default_sample_rate, WINDOW_SIZE as u32);

    let (sender, receiver) = channel();

    let mut circular_buffer = vec![0.0_f32; 4 * WINDOW_SIZE];
    let mut read_ptr = 0;
    let mut write_ptr = 0;

    let rotating_vectors = create_rotating_vectors(WINDOW_SIZE);
    let window = get_hann(WINDOW_SIZE);

    // TODO ensure signal increments by time frame
    let callback = move |pa::InputStreamCallbackArgs {buffer, ..}| {
        circular_buffer.splice(write_ptr..write_ptr + WINDOW_SIZE, buffer.iter().cloned());
        write_ptr = (write_ptr + WINDOW_SIZE) % circular_buffer.len();

        // only proceed if write pointer holds two windows from the read pointer, or
        // if the write pointer has wrapped around
        if write_ptr >= 2 * WINDOW_SIZE + read_ptr || write_ptr < read_ptr {
            let mut output_signal: Vec<Vec<f64>> = Vec::with_capacity(WINDOW_SIZE / N_OVERLAP);
            for i in 0..WINDOW_SIZE / N_OVERLAP {
                // read pointer should not exceed buffer length due to constraints from the for loop
                let mut signal: Vec<Float> = circular_read(&circular_buffer, read_ptr + i * N_OVERLAP, WINDOW_SIZE).iter().map(|x| *x as f64).collect();
                // apply the window to the output
                for (x, w) in signal.iter_mut().zip(&window) {
                    *x *= w;
                }

                iterative_rfft(&mut signal, &rotating_vectors);
                output_signal.push(signal);
            }

            read_ptr = (read_ptr + WINDOW_SIZE) % circular_buffer.len();

            match sender.send(output_signal) {
                Ok(_) => pa::Continue,
                Err(_) => pa::Complete,
            }
        } else {
            pa::Continue
        }
    };

    let mut stream = pa.open_non_blocking_stream(input_settings, callback)?;
    stream.start()?;

    while let true = stream.is_active()? {
        while let Ok(buffer) = receiver.try_recv() {
            for signal in buffer {
                let frequencies = get_frequencies(WINDOW_SIZE, (1.0 / 44100.0) as Float);
                let magnitudes: Vec<Float> = signal[signal.len() / 2 + 1..].iter().rev().zip(&signal[0..=signal.len() / 2]).map(|(re, im)| (re * re + im * im).sqrt()).collect();
            }
        }
    }

    stream.stop()?;

    Ok(())
}

