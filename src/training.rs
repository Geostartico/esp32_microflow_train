use std::sync::{Arc, Mutex};

use embedded_sdmmc::{BlockDevice, TimeSource, VolumeManager};
use esp_idf_sys::esp_random;
use nalgebra::{SMatrix, matrix};
use microflow::microflow_train_macros::model;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::read_image::open_image;

// #[model("models/outside_inside.tflite", 2, "crossentropy", true, [0.0], [1024.0])]
#[model("models/outside_inside.tflite", 2, "crossentropy", true, [0.0], [512.0])]
pub struct OutsideInsideModel {}
const VALIDATION_SPLIT: f32 = 0.2;
const OUTPUT_SCALE: f32 = 0.00390625;
const OUTPUT_ZERO_POINT: i8 = -128;
const EARLY_STOPPING_PATIENCE: usize = 3;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 3;

pub fn training_loop(
    volume_mgr: &VolumeManager<impl BlockDevice, impl TimeSource>,
    label_0_dir: embedded_sdmmc::RawDirectory,
    label_1_dir: embedded_sdmmc::RawDirectory,
    image_mutex: Arc<Mutex<Box<SMatrix<[f32; 3], 32, 32>>>>,
    model: &mut OutsideInsideModel,
    epochs: usize,
) {
    let mut rng = SmallRng::seed_from_u64((unsafe { esp_random() }) as u64);
    let mut labels_0 = vec![];
    let mut labels_1 = vec![];
    volume_mgr
        .iterate_dir(label_0_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels_0.push(entry.name.to_string());
            }
        })
        .ok();
    volume_mgr
        .iterate_dir(label_1_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels_1.push(entry.name.to_string());
            }
        })
        .ok();
    labels_0.shuffle(&mut rng);
    labels_1.shuffle(&mut rng);
    let validation_0 = labels_0.split_off(
        ((labels_0.len() as f32) * (1f32 - VALIDATION_SPLIT)).round() as usize
    );
    let validation_1 = labels_1.split_off(
        ((labels_1.len() as f32) * (1f32 - VALIDATION_SPLIT)).round() as usize
    );
    let mut train_vec: Vec<_> = labels_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(labels_1.into_iter().map(|x| (x, 1)))
        .collect();
    let validation_vec: Vec<_> = validation_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(validation_1.into_iter().map(|x| (x, 1)))
        .collect();
    let correct = validation_vec
        .iter()
        .map(|sample| {
            // print_stack_pointer();
            let mut unlocked = image_mutex.lock().unwrap();
            // log:info!("output: {}", output.buffer);
            let dir = if sample.1 == 0 { label_0_dir } else { label_1_dir };
            let now = unsafe { esp_idf_sys::esp_timer_get_time()};
            open_image(&volume_mgr, dir, &mut unlocked, &sample.0);
            let elapsed_time = unsafe { esp_idf_sys::esp_timer_get_time() } - now;
            log::info!("time to open image: {} us", elapsed_time);
            let now = unsafe { esp_idf_sys::esp_timer_get_time()};
            let result = model.predict([**unlocked]);
            let elapsed_time = unsafe { esp_idf_sys::esp_timer_get_time() } - now;
            log::info!("time to predict image: {} us", elapsed_time);
            unsafe {
                esp_idf_sys::vTaskDelay(5);
            }
            // println!("validation result: {},{}", result[0], result[1]);
            if sample.1 == 1 && result[1] > result[0] {
                1
            } else if sample.1 == 0 && result[0] > result[1] {
                1
            } else {
                0
            }
        })
        .reduce(|acc, val| acc + val)
        .unwrap();
    unsafe {
        esp_idf_sys::vTaskDelay(100);
    }
    log::info!("correct: {}/{}", correct, validation_vec.len());
    unsafe {
        esp_idf_sys::vTaskDelay(100);
    }
    train_vec.shuffle(&mut rng);
    let mut best_val = correct;
    let mut epochs_since_improvement = 0;
    for epoch in 0..epochs {
        let mut unlocked = image_mutex.lock().unwrap();
        let initial = model.weights0.buffer.clone().cast::<i32>();
        train_vec.shuffle(&mut rng);
        for (index, sample) in train_vec.iter().enumerate() {
            unsafe {
                esp_idf_sys::vTaskDelay(5);
            }
            let y = if sample.1 == 0 { matrix![1f32, 0f32] } else { matrix![0f32, 1f32] };
            let output = microflow::tensor::Tensor2D::quantize(
                y,
                [OUTPUT_SCALE],
                [OUTPUT_ZERO_POINT]
            );

            // log:info!("output: {}", output.buffer);
            let dir = if sample.1 == 0 { label_0_dir } else { label_1_dir };
            open_image(&volume_mgr, dir, &mut unlocked, &sample.0);
            log::info!("training on image {}", sample.0);
            let now = unsafe { esp_idf_sys::esp_timer_get_time()};
            let predicted_output = model.predict_train([**unlocked], &output, LEARNING_RATE);
            let elapsed_time = unsafe { esp_idf_sys::esp_timer_get_time() } - now;
            log::info!("time to predict_train image: {} us", elapsed_time);
            // log:info!(
            //     "predicted output: {}",
            //     microflow::tensor::Tensor2D::quantize(
            //         predicted_output,
            //         [output_scale],
            //         [output_zero_point]
            //     )
            //     .buffer
            // );
            // log:info!("gradient: {}", model.weights0_gradient.view((0, 0), (4, 2)));
            // panic!();
            if index != 0 && index % BATCH_SIZE == 0 {
                log::info!("batch: {}", index / BATCH_SIZE);
                model.update_layers(BATCH_SIZE, LEARNING_RATE);
                // log:info!("new bias: {}", model.constants0.0)
            }
        }
        model.update_layers(BATCH_SIZE, LEARNING_RATE);
        let correct = validation_vec
            .iter()
            .map(|sample| {
                unsafe {
                    esp_idf_sys::vTaskDelay(5);
                }
                let dir = if sample.1 == 0 { label_0_dir } else { label_1_dir };
                open_image(&volume_mgr, dir, &mut unlocked, sample.0.as_str());
                let result = model.predict([**unlocked]);
                // log:info!("result: {}, {}", result[0], result[1]);
                log::info!("validating on image {}", sample.0);
                // print_stack_pointer();
                if sample.1 == 1 && result[1] > result[0] {
                    1
                } else if sample.1 == 0 && result[0] > result[1] {
                    1
                } else {
                    0
                }
            })
            .reduce(|acc, val| acc + val)
            .unwrap();
        let fin = model.weights0.buffer.cast::<i32>();
        let diff = fin - initial;
        let changed = diff.map(|el| if el != 0 { 1 } else { 0 }).fold(0, |acc, el| acc + el);
        let saturated = model.weights0.buffer
            .map(|el| if el >= 126 || el <= -126 { 1 } else { 0 })
            .fold(0, |acc, el| acc + el);
        log::info!("saturated params {}", saturated);
        log::info!("changed params {}", changed);
        unsafe {
            esp_idf_sys::vTaskDelay(100);
        }
        log::info!("validation accuracy : {}/{}", correct, validation_vec.len());
        unsafe {
            esp_idf_sys::vTaskDelay(100);
        }
        log::info!("epoch {} complete", epoch);
        if correct > best_val {
            best_val = correct;
            epochs_since_improvement = 0;
        }
        else{
            epochs_since_improvement += 1;
        }
        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE {
            log::info!("Early stopping triggered at epoch {}", epoch);
            break;
        }
    }
}

pub fn validation_loop(
    volume_mgr: &VolumeManager<impl BlockDevice, impl TimeSource>,
    label_0_dir: embedded_sdmmc::RawDirectory,
    label_1_dir: embedded_sdmmc::RawDirectory,
    image_mutex: Arc<Mutex<Box<SMatrix<[f32; 3], 32, 32>>>>,
    model: &mut OutsideInsideModel,
) {
    let mut rng = SmallRng::seed_from_u64((unsafe { esp_random() }) as u64);
    let mut labels_0 = vec![];
    let mut labels_1 = vec![];
    volume_mgr
        .iterate_dir(label_0_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels_0.push(entry.name.to_string());
            }
        })
        .ok();
    volume_mgr
        .iterate_dir(label_1_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels_1.push(entry.name.to_string());
            }
        })
        .ok();
    labels_0.shuffle(&mut rng);
    labels_1.shuffle(&mut rng);
    let all: Vec<_> = labels_0
        .into_iter()
        .map(|x| (x, 0))
        .chain(labels_1.into_iter().map(|x| (x, 1)))
        .collect();
    let correct = all
        .iter()
        .map(|sample| {
            let mut unlocked = image_mutex.lock().unwrap();
            // log:info!("output: {}", output.buffer);
            let dir = if sample.1 == 0 { label_0_dir } else { label_1_dir };
            open_image(&volume_mgr, dir, &mut unlocked, &sample.0);
            let result = model.predict([**unlocked]);
            unsafe {
                esp_idf_sys::vTaskDelay(5);
            }
            // println!("validation result: {},{}", result[0], result[1]);
            if sample.1 == 1 && result[1] > result[0] {
                log::info!("predicted correct outside for {}; {},{}", sample.0, result[0], result[1]);
                1
            } else if sample.1 == 0 && result[0] > result[1] {
                log::info!("predicted correct inside for {}; {},{}", sample.0, result[0], result[1]);
                1
            } else {
                log::info!("predicted incorrect inside for {} with class {}; {},{}", sample.0, sample.1, result[0], result[1]);
                0
            }
        })
        .reduce(|acc, val| acc + val)
        .unwrap();
    unsafe {
        esp_idf_sys::vTaskDelay(100);
    }
    log::info!("correct validation: {}/{}", correct, all.len());
}

// fn print_stack_pointer() {
//     // Prefer a real inline-asm read of the SP when the optional feature is enabled
//     // and the target architecture is supported. Otherwise fall back to the
//     // address-of-local workaround which approximates the current stack pointer.
//     //
//     // To enable the inline asm path, build with:
//     //   cargo build --features use-inline-asm
//     //
//     // Note: inline asm requires a toolchain that supports it for your target.
//     #[cfg(all(feature = "use-inline-asm"))]
//     {
//         let sp: usize;
//         unsafe {
//             core::arch::asm!("mov {0}, a1", out(reg) sp);
//         }
//         log::info!("Stack Pointer : {:#X}", sp);
//         return;
//     }
// }