use std::sync::Arc;
use std::sync::Mutex;

use nalgebra::SMatrix;
use embedded_sdmmc::BlockDevice;
use embedded_sdmmc::TimeSource;
use embedded_sdmmc::{ SdCard, VolumeManager, VolumeIdx, Mode };

use crate::camera::Camera_wrapper;

const IMAGES: usize = 200;

pub fn downsample_rgb565(
    input: &[u8],
    src_w: usize,
    src_h: usize,
    target_w: usize,
    target_h: usize,
    image_mat: &mut SMatrix<[f32; 3], 32, 32>
) {
    let scale_x = (src_w as f32) / (target_w as f32);
    let scale_y = (src_h as f32) / (target_h as f32);

    for y in 0..target_h {
        for x in 0..target_w {
            let src_x_f = (x as f32) * scale_x;
            let src_y_f = (y as f32) * scale_y;

            let x0 = src_x_f as usize;
            let y0 = src_y_f as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = src_x_f - (x0 as f32);
            let fy = src_y_f - (y0 as f32);

            // Get four neighboring pixels - READ BYTES CORRECTLY
            let p00 = read_rgb565_pixel_be(input, y0 * src_w + x0);
            let p10 = read_rgb565_pixel_be(input, y0 * src_w + x1);
            let p01 = read_rgb565_pixel_be(input, y1 * src_w + x0);
            let p11 = read_rgb565_pixel_be(input, y1 * src_w + x1);

            image_mat[(x, y)] = bilinear_rgb565_f32(p00, p10, p01, p11, fx, fy);
        }
    }
}

/// Read a single RGB565 pixel with correct byte order
/// ESP32 camera typically outputs in little-endian format
#[inline]
pub fn read_rgb565_pixel(data: &[u8], pixel_index: usize) -> u16 {
    let byte_index = pixel_index * 2;

    // Read as little-endian: low byte first, then high byte
    let low = data[byte_index] as u16;
    let high = data[byte_index + 1] as u16;

    // Combine: high byte in upper 8 bits, low byte in lower 8 bits
    (high << 8) | low
}

/// Alternative: Read as big-endian if the above doesn't work
#[inline]
pub fn read_rgb565_pixel_be(data: &[u8], pixel_index: usize) -> u16 {
    let byte_index = pixel_index * 2;

    // Read as big-endian: high byte first, then low byte
    let high = data[byte_index] as u16;
    let low = data[byte_index + 1] as u16;

    (high << 8) | low
}

pub fn bilinear_rgb565_f32(p00: u16, p10: u16, p01: u16, p11: u16, fx: f32, fy: f32) -> [f32; 3] {
    // Extract RGB from RGB565: RRRRRGGGGGGBBBBB
    let r00 = (((p00 >> 11) & 0x1f) as f32) / 31.0;
    let g00 = (((p00 >> 5) & 0x3f) as f32) / 63.0;
    let b00 = ((p00 & 0x1f) as f32) / 31.0;

    let r10 = (((p10 >> 11) & 0x1f) as f32) / 31.0;
    let g10 = (((p10 >> 5) & 0x3f) as f32) / 63.0;
    let b10 = ((p10 & 0x1f) as f32) / 31.0;

    let r01 = (((p01 >> 11) & 0x1f) as f32) / 31.0;
    let g01 = (((p01 >> 5) & 0x3f) as f32) / 63.0;
    let b01 = ((p01 & 0x1f) as f32) / 31.0;

    let r11 = (((p11 >> 11) & 0x1f) as f32) / 31.0;
    let g11 = (((p11 >> 5) & 0x3f) as f32) / 63.0;
    let b11 = ((p11 & 0x1f) as f32) / 31.0;

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let r = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
    let g = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11;
    let b = b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11;

    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

/// Debug function to test byte order - call this first!
pub fn test_rgb565_byte_order(data: &[u8]) {
    if data.len() < 20 {
        log::error!("Not enough data to test");
        return;
    }

    log::info!("Testing RGB565 byte order...");
    log::info!("First 10 bytes: {:02X?}", &data[0..10]);

    // Test little-endian interpretation
    for i in 0..5 {
        let pixel_le = read_rgb565_pixel(data, i);
        let r_le = ((pixel_le >> 11) & 0x1f) as u8;
        let g_le = ((pixel_le >> 5) & 0x3f) as u8;
        let b_le = (pixel_le & 0x1f) as u8;

        // Convert to 8-bit for display
        let r8_le = (r_le << 3) | (r_le >> 2);
        let g8_le = (g_le << 2) | (g_le >> 4);
        let b8_le = (b_le << 3) | (b_le >> 2);

        log::info!("Pixel {} LE: 0x{:04X} -> R={} G={} B={}", i, pixel_le, r8_le, g8_le, b8_le);
    }

    log::info!("---");

    // Test big-endian interpretation
    for i in 0..5 {
        let pixel_be = read_rgb565_pixel_be(data, i);
        let r_be = ((pixel_be >> 11) & 0x1f) as u8;
        let g_be = ((pixel_be >> 5) & 0x3f) as u8;
        let b_be = (pixel_be & 0x1f) as u8;

        let r8_be = (r_be << 3) | (r_be >> 2);
        let g8_be = (g_be << 2) | (g_be >> 4);
        let b8_be = (b_be << 3) | (b_be >> 2);

        log::info!("Pixel {} BE: 0x{:04X} -> R={} G={} B={}", i, pixel_be, r8_be, g8_be, b8_be);
    }

    log::info!("Look at the RGB values above. Which interpretation looks more reasonable?");
    log::info!("If LE values look correct, use read_rgb565_pixel()");
    log::info!("If BE values look correct, use read_rgb565_pixel_be()");
}

/// Fixed version using big-endian if that's what's needed
pub fn downsample_rgb565_be(
    input: &[u8],
    src_w: usize,
    src_h: usize,
    target_w: usize,
    target_h: usize,
    image_mat: &mut SMatrix<[f32; 3], 32, 32>
) {
    let scale_x = (src_w as f32) / (target_w as f32);
    let scale_y = (src_h as f32) / (target_h as f32);

    for y in 0..target_h {
        for x in 0..target_w {
            let src_x_f = (x as f32) * scale_x;
            let src_y_f = (y as f32) * scale_y;

            let x0 = src_x_f as usize;
            let y0 = src_y_f as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = src_x_f - (x0 as f32);
            let fy = src_y_f - (y0 as f32);

            // Use big-endian reading
            let p00 = read_rgb565_pixel_be(input, y0 * src_w + x0);
            let p10 = read_rgb565_pixel_be(input, y0 * src_w + x1);
            let p01 = read_rgb565_pixel_be(input, y1 * src_w + x0);
            let p11 = read_rgb565_pixel_be(input, y1 * src_w + x1);

            image_mat[(x, y)] = bilinear_rgb565_f32(p00, p10, p01, p11, fx, fy);
        }
    }
}

pub fn save_image<'a, R: BlockDevice, S: TimeSource>(
    volume_mgr: &VolumeManager<R, S>,
    images_dir: embedded_sdmmc::RawDirectory,
    photos_counter: usize,
    image_mat: &SMatrix<[f32; 3], 32, 32>
) {
    let file_name = format!("ph_{}", photos_counter);

    // Open file for writing
    let mut file = volume_mgr
        .open_file_in_dir(images_dir, file_name.as_str(), Mode::ReadWriteCreateOrTruncate)
        .unwrap()
        .to_file(&volume_mgr);

    // Convert to u8 (0-255 range)
    let mut buffer = Vec::with_capacity(32 * 32 * 3);

    for el in image_mat.as_slice().iter() {
        for fl in el.iter() {
            let byte = (fl.clamp(0.0, 1.0) * 255.0) as u8;
            buffer.push(byte);
        }
    }

    // Write the buffer
    match file.write(&buffer) {
        Ok(()) => {
            log::info!("Successfully wrote {} bytes to {}", buffer.len(), file_name);
        }
        Err(e) => {
            log::error!("Failed to write image {}: {:?}", file_name, e);
        }
    }

    // CRITICAL: Close the file to ensure data is flushed to SD card
    drop(file);

    // Optional: Add a small delay to ensure write completion
    unsafe {
        esp_idf_sys::vTaskDelay(2);
    }
}

pub fn open_image<'a, R: BlockDevice, S: TimeSource>(
    volume_mgr: &VolumeManager<R, S>,
    images_dir: embedded_sdmmc::RawDirectory,
    image_mat: &mut SMatrix<[f32; 3], 32, 32>,
    file_name: &str
) {
    // Open file for reading only
    let mut file = match volume_mgr.open_file_in_dir(images_dir, file_name, Mode::ReadOnly) {
        Ok(f) => f.to_file(&volume_mgr),
        Err(e) => {
            log::error!("Failed to open file {}: {:?}", file_name, e);
            return;
        }
    };

    let mut buffer = [0u8; 32 * 32 * 3];

    // Read the data
    match file.read(&mut buffer) {
        Ok(read) => {
            log::info!("Successfully read {} bytes from {}", read, file_name);
        }
        Err(e) => {
            log::error!("Failed to read image {}: {:?}", file_name, e);
            drop(file);
            return;
        }
    }

    // Convert buffer back to float matrix
    for (i, chunk) in buffer.chunks_exact(3).enumerate() {
        let x = i / 32;
        let y = i % 32;
        image_mat[(x, y)] = [
            (chunk[0] as f32) / 255.0,
            (chunk[1] as f32) / 255.0,
            (chunk[2] as f32) / 255.0,
        ];
    }

    // Close the file
    drop(file);
}

// Verification helper function
pub fn verify_image_saved<'a, R: BlockDevice, S: TimeSource>(
    volume_mgr: &VolumeManager<R, S>,
    images_dir: embedded_sdmmc::RawDirectory,
    file_name: &str
) -> bool {
    // Try to read the file to verify it exists and has correct size
    match volume_mgr.open_file_in_dir(images_dir, file_name, Mode::ReadOnly) {
        Ok(file) => {
            let mut test_buffer = [0u8; 3072];
            let mut file_handle = file.to_file(&volume_mgr);
            match file_handle.read(&mut test_buffer) {
                Ok(read) => {
                    if read != 3072 {
                        log::error!("File {} has incorrect size: {} bytes", file_name, read);
                        drop(file_handle);
                        return false;
                    }
                    log::info!("File {} verified successfully (3072 bytes)", file_name);
                    drop(file_handle);
                    true
                }
                Err(e) => {
                    log::error!("Failed to read {} during verification: {:?}", file_name, e);
                    drop(file_handle);
                    false
                }
            }
        }
        Err(e) => {
            log::error!("Failed to open {} for verification: {:?}", file_name, e);
            false
        }
    }
}

pub fn save_bulk_images_and_take_commands(
    camera: &Camera_wrapper,
    volume_mgr: &VolumeManager<impl BlockDevice, impl TimeSource>,
    dir: embedded_sdmmc::RawDirectory,
    mutex_loop: Arc<Mutex<Box<SMatrix<[f32; 3], 32, 32>>>>,
    command_loop: Arc<Mutex<[u8; 1]>>
) {
    let mut photos_counter = 0;;
    let mut take_pictures = false;
    loop {
        unsafe {
            esp_idf_sys::vTaskDelay(100);
        }
        if take_pictures {
            log::info!("Getting framebuffer...");
            let framebuffer = camera.get_framebuffer();
            log::info!("Framebuffer obtained.");

            if let Some(framebuffer) = framebuffer {
                //TODO: put back the original dir
                let data = framebuffer.data();
                log::info!("sampling image...");
                match mutex_loop.try_lock() {
                    Ok(mut unlocked_image) => {
                        downsample_rgb565(
                            data,
                            framebuffer.width() as usize,
                            framebuffer.height() as usize,
                            32,
                            32,
                            &mut *&mut unlocked_image
                        );

                        log::info!("saving image {}... ", photos_counter);
                        save_image(&volume_mgr, dir, photos_counter, &unlocked_image);
                        photos_counter += 1;
                        log::info!("image saved.");
                    }
                    Err(_) => {
                        log::info!("no framebuffer");
                    }
                }
            }
        }
        if photos_counter >= IMAGES {
            break;
        }
        match command_loop.try_lock() {
            Ok(mut unlocked_command) => {
                if unlocked_command[0] != 0 {
                    let command_str = match unlocked_command[0] {
                        1 => "up",
                        2 => "down",
                        3 => "left",
                        4 => "right",
                        5 => "stop",
                        6 => {
                            take_pictures = true;
                            unsafe {
                                esp_idf_sys::vTaskDelay(100);
                            }
                            log::info!("\r\nup\r\n");
                            unsafe {
                                esp_idf_sys::vTaskDelay(100);
                            }
                            ""
                        }
                        _ => "",
                    };
                    if !command_str.is_empty() {
                        log::info!("\r\n{}\r\n", command_str);
                    }
                    unlocked_command[0] = 0;
                }
            }
            Err(_) => {/* Handle lock error if necessary */}
        }
    }
    unsafe {
        esp_idf_sys::vTaskDelay(100);
    }
    log::info!("\r\nstop\r\n");
    unsafe {
        esp_idf_sys::vTaskDelay(100);
    }
    let framebuffer = camera.get_framebuffer();
}