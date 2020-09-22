use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use js_sys::Float32Array;

use std::cmp;

use fft::fft::*;
use complex::complex::{Float, Complex};

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    console::log_1(&JsValue::from_str("Hello, World!"));

    Ok(())
}

struct CircularBuffer<T>
where
    T: Clone,
{
    buffer: Vec<T>,
    read_ptr: usize,
    write_ptr: usize,
}

impl<T> CircularBuffer<T>
where
    T: Clone,
{
    fn read(&mut self, read_offset: usize, read_len: usize) -> Vec<T> {
        let buffer_length = self.buffer.len();
        let offset: i32 = (self.read_ptr + read_offset + read_len) as i32 - buffer_length as i32;
        let end_ptr = cmp::min(self.read_ptr + read_offset + read_len, buffer_length);

        let mut result = self.buffer[self.read_ptr + read_offset..end_ptr].to_vec();

        if offset > 0 {
            result.extend(self.buffer[0..offset as usize].to_vec());
        }

        result
    }

    fn append(&mut self, signal: &Vec<T>) {
        self.buffer.splice(self.write_ptr..self.write_ptr + signal.len(), signal.iter().cloned());
        self.write_ptr = (self.write_ptr + signal.len()) % self.buffer.len();
    }
}

fn get_document() -> web_sys::Document {
    web_sys::window().unwrap().document().unwrap()
}

fn get_canvas(id: &str) -> web_sys::HtmlCanvasElement {
    let canvas = get_document().get_element_by_id(id).unwrap();
    canvas.dyn_into::<web_sys::HtmlCanvasElement>().map_err(|_| ()).unwrap()
}

fn get_context(canvas: &web_sys::HtmlCanvasElement) -> web_sys::CanvasRenderingContext2d {
    canvas.get_context_with_context_options("2d", &JsValue::from_str("{ alpha: false }")).unwrap().unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap()
}

#[wasm_bindgen]
pub struct Spectrogram {
    circular_buffer: CircularBuffer<Float>,
    overlap: usize,
    rotating_vectors: Vec<Complex>,
    hann_window: Vec<Float>,
    block_height: f64,
}

#[wasm_bindgen]
impl Spectrogram {
    #[wasm_bindgen(constructor)]
    pub fn new(buffer_length: usize) -> Self {
        let circular_buffer = CircularBuffer {
            buffer: vec![0.0 as Float; 4 * buffer_length],
            read_ptr: 0,
            write_ptr: 0,
        };

        let overlap = buffer_length / 4;

        let rotating_vectors = create_rotating_vectors(buffer_length);
        let hann_window = create_hann(buffer_length);
        let frequencies = get_real_frequencies(buffer_length, (1.0 / 44100.0) as Float);

        let canvas = get_canvas("canvas");
        let ctx = get_context(&canvas);

        let grid_canvas = get_canvas("grids");
        let grid_ctx = get_context(&grid_canvas);

        canvas.set_width(8 * canvas.width());
        canvas.set_height(2 * canvas.height());

        grid_canvas.set_width(2 * grid_canvas.width());
        grid_canvas.set_height(2 * grid_canvas.height());

        ctx.set_image_smoothing_enabled(false);
        grid_ctx.set_image_smoothing_enabled(false);

        grid_ctx.set_font("7px sans-serif");
        grid_ctx.set_fill_style(&JsValue::from_str("rgb(0, 255, 0)"));
        grid_ctx.set_stroke_style(&JsValue::from_str("rgb(0, 255, 0)"));
        grid_ctx.set_line_width(0.25);

        let max_freq = frequencies[frequencies.len() - 1];
        let num_steps = 10;
        let step = grid_canvas.height() as usize / num_steps;
        for y in (step..grid_canvas.height() as usize).step_by(step) {
            grid_ctx.fill_text(&format!("{:.0}", (num_steps - y / step) as Float / num_steps as Float * max_freq), 0.5, y as f64 - 0.5).expect("Failed to write text to canvas.");
            grid_ctx.move_to(0.5, y as f64 - 0.5);
            grid_ctx.line_to(grid_canvas.width() as f64 - 0.5, y as f64 - 0.5);
        }

        grid_ctx.stroke();

        // canvas.set_width(web_sys::window().unwrap().inner_width().unwrap().as_f64().unwrap() as u32);
        // canvas.set_height(web_sys::window().unwrap().inner_height().unwrap().as_f64().unwrap() as u32);

        let block_height = 2.0 * canvas.height() as f64 / buffer_length as f64;

        Self {
            circular_buffer,
            overlap,
            rotating_vectors,
            hann_window,
            block_height,
        }
    }

    #[wasm_bindgen]
    pub fn process_signal(&mut self, signal: Float32Array) {
        let circular_buffer = &mut self.circular_buffer;
        let (read_ptr, write_ptr) = (circular_buffer.read_ptr, circular_buffer.write_ptr);

        let signal: Vec<Float> = signal.to_vec().into_iter().map(|x| x as Float).collect();
        let signal_length = signal.len();
        circular_buffer.append(&signal);

        let canvas = get_canvas("canvas");
        let ctx = get_context(&canvas);

        let (height, width) = (canvas.height(), canvas.width());

        let block_height = self.block_height;

        // only proceed if write pointer is two windows from the read pointer
        if (write_ptr as i32 - read_ptr as i32).abs() >= (2 * signal_length) as i32 {
            let mut output_signal: Vec<Vec<Float>> = Vec::with_capacity(signal_length / self.overlap);
            
            for i in 0..signal_length / self.overlap {
                // read pointer should not exceed buffer length due to constraints from the for loop
                let mut signal: Vec<Float> = circular_buffer.read(i * self.overlap, signal_length).iter().map(|x| *x as Float).collect();
                
                // apply the window to the output
                for (x, w) in signal.iter_mut().zip(&self.hann_window) {
                    *x *= w;
                }

                iterative_rfft(&mut signal, &self.rotating_vectors);

                let mut magnitudes = vec![0.0; signal_length / 2 + 1];

                // real-only values
                magnitudes[0] = signal[0];
                magnitudes[signal_length / 2] = signal[signal_length / 2];

                for ((im, re), magnitude) in signal[signal_length / 2 + 1..].iter().rev().zip(&signal[1..signal_length / 2]).zip(&mut magnitudes[1..signal_length / 2]) {
                    *magnitude = (6.0 * re * re + 6.0 * im * im).sqrt();
                }

                output_signal.push(magnitudes);
            }

            circular_buffer.read_ptr = (read_ptr + signal.len()) % circular_buffer.buffer.len();

            let overlap = self.overlap;
            let f = Closure::wrap(Box::new(move || {
                let canvas = get_canvas("canvas");

                let num_windows = signal_length / overlap;
                let mut pixel_data = vec![0; num_windows * height as usize * 4];
                for i in 0..num_windows {
                    let magnitudes: Vec<(usize, &f64)> = output_signal[i].iter().enumerate().filter(|&(_, x)| *x > 0.3).collect();
                    let max = magnitudes.iter().cloned().fold(0.0 / 0.0, |m, (_, x)| f64::max(m, *x));
                    for (j, x) in magnitudes {
                        let y = (height as f64 - j as f64 * block_height - 0.5).floor() as usize;
                        let intensity = (2.0 + (x / max) * 230.0) as u8;
                        // add a slight offset to the colours
                        // ctx.set_fill_style(&JsValue::from_str(&format!("rgb({}, {}, {})", 0, intensity, 0)));
                        // ctx.fill_rect(width as f64 - 1.5, y, 1.0, block_height);
                        let position = (num_windows as usize * y + i) * 4;
                        pixel_data[position + 1] = intensity;
                        pixel_data[position + 3] = 255;
                    } 
                }

                ctx.set_global_composite_operation("copy").expect("Failed to change global composite operation.");
                ctx.draw_image_with_html_canvas_element(&canvas, -(num_windows as f64), 0.0).expect("Failed to draw canvas image.");
                ctx.set_global_composite_operation("source-over").expect("Failed to change global composite operation.");


                let image_data = web_sys::ImageData::new_with_u8_clamped_array(wasm_bindgen::Clamped(&mut pixel_data), num_windows as u32).unwrap();
                ctx.put_image_data(&image_data, width as f64 - num_windows as f64 - 0.5, 0.0).expect("Failed to write image data.");
            }) as Box<dyn FnMut()>);

            web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).expect("Failed to request animation frame.");
            f.forget();
        }
    }
}

