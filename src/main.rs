use portaudio as pa;
use std::sync::mpsc::*;

use fft::fft::*;

const WINDOW_SIZE: usize = 256;

fn main() {
    match run() {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run PortAudio with: {:?}", e)
        }
    }
}

fn run() -> Result<(), pa::Error> {
    let pa = pa::PortAudio::new()?;
    let mic_index = pa.default_input_device()?;
    let mic = pa.device_info(mic_index)?;

    let input_params = pa::StreamParameters::<f32>::new(mic_index, 1, true, mic.default_low_input_latency);

    let input_settings = pa::InputStreamSettings::new(input_params, mic.default_sample_rate, WINDOW_SIZE as u32);

    let (sender, receiver) = channel();

    let mut circular_buffer = vec![0.0_f32; 4 * WINDOW_SIZE];
    let mut read = 0;
    let mut write = 0;

    let rotating_vectors = create_rotating_vectors(WINDOW_SIZE);

    // TODO ensure signal increments by time frame
    let callback = move |pa::InputStreamCallbackArgs {buffer, ..}| {
        circular_buffer.splice(write..write + WINDOW_SIZE, buffer.iter().cloned());
        write = (write + WINDOW_SIZE) % circular_buffer.len();

        // TODO overlap is required due to windowing function
        let mut output_signal = circular_buffer[read..read + WINDOW_SIZE].iter().map(|x| *x as f64).collect();
        iterative_hann(&mut output_signal);
        iterative_rfft(&mut output_signal, &rotating_vectors);

        read = (read + WINDOW_SIZE) % circular_buffer.len();

        match sender.send(output_signal) {
            Ok(_) => pa::Continue,
            Err(_) => pa::Complete,
        }
    };

    let mut stream = pa.open_non_blocking_stream(input_settings, callback)?;
    stream.start()?;

    while let true = stream.is_active()? {
        while let Ok(buffer) = receiver.try_recv() {
            println!("{:?}", buffer);
        }
    }

    stream.stop()?;

    Ok(())
}

