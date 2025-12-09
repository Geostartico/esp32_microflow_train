#![feature(asm_experimental_arch)]

use bytemuck::cast_slice;
use embedded_sdmmc::BlockDevice;
use embedded_sdmmc::TimeSource;
use esp_idf_hal::spi::config::DriverConfig;
use esp_idf_sys::speed_t;
use esp32_cam_microflow::camera::*;
use esp_idf_hal::modem::Modem;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::http::server::EspHttpServer;
use esp_idf_svc::http::Method;
use esp_idf_svc::io::EspIOError;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::wifi::BlockingWifi;
use esp_idf_svc::wifi::EspWifi;
use esp_idf_svc::wifi::{ AuthMethod, ClientConfiguration, Configuration };
use esp_idf_sys::camera::{ esp_camera_sensor_get, exit };
use microflow::buffer::Buffer2D;
use core::panic;
use std::arch::asm;
use std::collections::HashMap;
use std::sync::{ Arc, Mutex };
use microflow::microflow_train_macros::model;
use nalgebra::{ SMatrix, matrix };
use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::spi::{ SpiDeviceDriver, SpiDriver, config::Config as SpiConfig };
use esp_idf_hal::units::Hertz;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use embedded_sdmmc::{ SdCard, VolumeManager, VolumeIdx, Mode };
// #[model("models/outside_inside.tflite", 2, "crossentropy", true, [0.0], [1024.0])]
#[model("models/outside_inside.tflite", 2, "crossentropy", true, [0.0], [512.0])]
struct OutsideInsideModel {}
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f32 = 0.01;
const IMAGES: usize = 200;
const EPOCHS: usize = 3;
const VALIDATION_SPLIT: f32 = 0.2;
const OUTPUT_SCALE: f32 = 0.00390625;
const OUTPUT_ZERO_POINT: i8 = -128;
const EARLY_STOPPING_PATIENCE: usize = 3;

extern "C" {
    fn esp_random() -> u32;
}
// Helper function to parse query parameters from URI
fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '%' => {
                // Get next two hex digits
                let hex: String = chars.by_ref().take(2).collect();
                if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                    result.push(byte as char);
                } else {
                    result.push('%');
                    result.push_str(&hex);
                }
            }
            '+' => result.push(' '),
            _ => result.push(c),
        }
    }

    result
}

fn parse_query_params(uri: &str) -> Option<HashMap<String, String>> {
    // Find the query string after '?'
    let query_start = uri.find('?')?;
    let query_string = &uri[query_start + 1..];

    // Handle empty query string
    if query_string.is_empty() {
        return None;
    }

    let mut params = HashMap::new();

    // Split by '&' and parse each key=value pair
    for param in query_string.split('&') {
        if let Some((key, value)) = param.split_once('=') {
            // URL decode the values
            let decoded_key = url_decode(key);
            let decoded_value = url_decode(value);
            params.insert(decoded_key, decoded_value);
        }
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}
fn connect_wifi<'a>(ssid: &str, password: &str, modem: Modem) -> BlockingWifi<EspWifi<'a>> {
    // Initialize WiFi
    log::info!("taking event loop...");
    let sys_loop = EspSystemEventLoop::take().unwrap();
    log::info!("taking nvs...");
    let nvs = EspDefaultNvsPartition::take().unwrap();
    log::info!("initializing blocking wifi...");
    let mut wifi = BlockingWifi::wrap(
        EspWifi::new(modem, sys_loop.clone(), Some(nvs)).unwrap(),
        sys_loop
    ).unwrap();

    log::info!("starting wifi for scanning...");
    // Start WiFi to enable scanning
    wifi.start().unwrap();

    log::info!("configuring...");
    // Configure WiFi
    wifi.set_configuration(
        &Configuration::Client(ClientConfiguration {
            ssid: ssid.try_into().unwrap(),
            bssid: None,
            auth_method: AuthMethod::WPA2Personal,
            password: password.try_into().unwrap(),
            channel: None,
            scan_method: esp_idf_svc::wifi::ScanMethod::FastScan,
            pmf_cfg: esp_idf_svc::wifi::PmfConfiguration::NotCapable,
        })
    ).unwrap();
    log::info!("scanning for available networks...");
    // Perform WiFi scan
    let scan_result = wifi.scan().unwrap();

    log::info!("Available SSIDs: {}", scan_result.len());
    for ap in scan_result.iter() {
        log::info!(
            "  SSID: {:?}, Signal: {} dBm, Channel: {}, Auth: {:?}",
            ap.ssid,
            ap.signal_strength,
            ap.channel,
            ap.auth_method
        );
    }

    log::info!("connecting to specified network...");
    // Connect to WiFi
    wifi.connect().unwrap();
    log::info!("WiFi connected");
    wifi.wait_netif_up().unwrap();
    log::info!("WiFi netif up");

    // Get and display IP address
    let ip_info = wifi.wifi().sta_netif().get_ip_info().unwrap();
    log::info!("IP Address: {:?}", ip_info.ip);
    log::info!("Camera server available at: http://{}/camera.jpg", ip_info.ip);
    wifi
}
/// Dummy time source for read-only operations
struct DummyTimeSource;

impl embedded_sdmmc::TimeSource for DummyTimeSource {
    fn get_timestamp(&self) -> embedded_sdmmc::Timestamp {
        embedded_sdmmc::Timestamp {
            year_since_1970: 0,
            zero_indexed_month: 0,
            zero_indexed_day: 0,
            hours: 0,
            minutes: 0,
            seconds: 0,
        }
    }
}

struct Delay;

impl embedded_hal::delay::DelayNs for Delay {
    fn delay_ns(&mut self, ns: u32) {
        let ms = ns / 1_000_000;
        if ms > 0 {
            FreeRtos::delay_ms(ms);
        }
    }
}

fn main() {
    // It is necessary to call this function once. Otherwise some patches to the runtime
    // implemented by esp-idf-sys might not link properly. See https://github.com/esp-rs/esp-idf-template/issues/71
    esp_idf_svc::sys::link_patches();

    // Bind the log crate to the ESP Logging facilities
    esp_idf_svc::log::EspLogger::initialize_default();
    log::info!("Hemlo, world!");
    #[cfg(feature = "use-inline-asm")]
    {
        print_stack_pointer();
    }
    let peripherals = Peripherals::take().unwrap();
    log::info!("ESP32 SD Card Reader with ESP-IDF");

    // Configure SPI pins
    let sclk = peripherals.pins.gpio14;
    let miso = peripherals.pins.gpio2;
    let mosi = peripherals.pins.gpio15;
    let cs = peripherals.pins.gpio13;

    // Create CS pin

    // Configure SPI driver
    let spi_config = DriverConfig::default();
    let spi_driver = SpiDriver::new(peripherals.spi2, sclk, mosi, Some(miso), &spi_config).unwrap();

    // Create SPI device
    let spi_config = SpiConfig::new().baudrate(Hertz(4_000_000));
    let spi_device = SpiDeviceDriver::new(spi_driver, Some(cs), &spi_config).unwrap();

    let delay = Delay;

    // Initialize SD card
    log::info!("Initializing SD card...");
    let sd_card = SdCard::new(spi_device, delay);

    // Get card size
    match sd_card.num_bytes() {
        Ok(size) => {
            log::info!("SD card size: {} bytes ({} MB)", size, size / (1024 * 1024));
        }
        Err(e) => {
            log::info!("Failed to get card size: {:?}", e);
        }
    }

    // Create volume manager
    let volume_mgr = VolumeManager::new(sd_card, DummyTimeSource);

    // Open volume
    log::info!("Opening volume...");
    let volume = match volume_mgr.open_volume(VolumeIdx(0)) {
        Ok(v) => {
            log::info!("Volume opened successfully!");
            v
        }
        Err(e) => {
            log::info!("Failed to open volume: {:?}", e);
            panic!("Failed to open volume");
        }
    };

    // Open root directory
    log::info!("Opening root directory...");
    let root_dir = match volume_mgr.open_root_dir(volume.to_raw_volume()) {
        Ok(dir) => {
            log::info!("Root directory opened!");
            dir
        }
        Err(e) => {
            log::info!("Failed to open root directory: {:?}", e);
            panic!("Failed to open root directory");
        }
    };

    // List files in root directory
    // log::info!("\nListing files:");
    // let mut dirs = vec![];
    // volume_mgr
    //     .iterate_dir(root_dir, |entry| {
    //         log::info!("  - {} ({}{})", entry.name, entry.size, if entry.attributes.is_directory() {
    //             " DIR"
    //         } else {
    //             ""
    //         });
    //         if entry.attributes.is_directory() {
    //             dirs.push(entry.name.clone());
    //         }
    //     })
    //     .ok();
    // for dir_name in dirs {
    //     log::info!("Entering directory: {}", dir_name);
    //     if let Ok(sub_dir) = volume_mgr.open_dir(root_dir, &dir_name) {
    //         volume_mgr
    //             .iterate_dir(sub_dir, |entry| {
    //                 log::info!("  - {} ({}{})", entry.name, entry.size, if
    //                     entry.attributes.is_directory()
    //                 {
    //                     " DIR"
    //                 } else {
    //                     ""
    //                 });
    //             })
    //             .ok();
    //     }
    // }
    log::info!("connecting to wifi");
    let _wifi = connect_wifi("comicsans", "helloooo", peripherals.modem);
    log::info!("connected to wifi");
    // let _led = PinDriver::output(peripherals.pins.gpio33).unwrap();
    let camera = Camera_wrapper::new(
        peripherals.pins.gpio32,
        peripherals.pins.gpio0,
        peripherals.pins.gpio5,
        peripherals.pins.gpio18,
        peripherals.pins.gpio19,
        peripherals.pins.gpio21,
        peripherals.pins.gpio36,
        peripherals.pins.gpio39,
        peripherals.pins.gpio34,
        peripherals.pins.gpio35,
        peripherals.pins.gpio25,
        peripherals.pins.gpio23,
        peripherals.pins.gpio22,
        peripherals.pins.gpio26,
        peripherals.pins.gpio27,
        esp_idf_sys::camera::pixformat_t_PIXFORMAT_RGB565,
        esp_idf_sys::camera::framesize_t_FRAMESIZE_96X96
    ).unwrap_or_else(|_el| {
        log::info!("Failed to initialize camera");
        unsafe { exit(0) }
    });

    log::info!("Camera initialized");
    let camera_sensor = unsafe { esp_camera_sensor_get() };
    log::info!("Camera sensor obtained");
    let camera_sensor = CameraSensor::new(camera_sensor);
    camera_sensor.set_vflip(true).unwrap();
    camera_sensor.set_brightness(2).unwrap();
    camera_sensor.set_saturation(2).unwrap();
    log::info!("Camera sensor vflip set to true");
    log::info!("Starting application...");
    let image = Box::new(Buffer2D::from_element([0f32; 3]));
    let image_mutex: Arc<Mutex<Box<SMatrix<[f32; 3], 32, 32>>>> = Arc::new(Mutex::new(image));
    let mutex_http = image_mutex.clone();
    let mutex_loop = image_mutex.clone();
    log::info!("Image matrix created");

    let command = Arc::new(Mutex::new([0u8; 1]));
    let command_call = command.clone();
    let command_loop = command.clone();
    let predict_train = Arc::new(Mutex::new([0u8; 1]));
    let pt_http = predict_train.clone();
    log::info!("uart created");
    let mut config = esp_idf_svc::http::server::Configuration::default();
    config.stack_size = 100000;
    let mut server = EspHttpServer::new(&config).unwrap();

    server
        .fn_handler("/move", Method::Get, move |request| {
            if let Some(params) = parse_query_params(request.uri()) {
                let action = params.get("action").map(|s| s.as_str());
                match action {
                    Some("up") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 1;
                                request.into_ok_response()?.write(format!("moving up").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not moving up").as_bytes())?;
                            }
                        }
                    }
                    Some("down") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 2;
                                request
                                    .into_status_response(500)?
                                    .write(format!("moving down").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not moving down").as_bytes())?;
                            }
                        }
                    }
                    Some("left") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 3;
                                request
                                    .into_status_response(500)?
                                    .write(format!("moving left").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not moving left").as_bytes())?;
                            }
                        }
                    }
                    Some("right") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 4;
                                request
                                    .into_ok_response()?
                                    .write(format!("moving right").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not moving right").as_bytes())?;
                            }
                        }
                    }
                    Some("stop") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 5;
                                request.into_ok_response()?.write(format!("stopping").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not stopping").as_bytes())?;
                            }
                        }
                    }
                    Some("start_readings") => {
                        match command_call.try_lock() {
                            Ok(mut unlocked) => {
                                unlocked[0] = 6;
                                request
                                    .into_ok_response()?
                                    .write(format!("start readings").as_bytes())?;
                            }
                            Err(_) => {
                                request
                                    .into_status_response(500)?
                                    .write(format!("not start readings").as_bytes())?;
                            }
                        }
                    }
                    _ => {
                        request.into_response(400, None, &[])?.write(b"Invalid parameters")?;
                    }
                }
            } else {
                request.into_response(400, None, &[])?.write(b"Missing parameters")?;
            }

            Ok::<(), EspIOError>(())
        })
        .unwrap();
    log::info!("Initializing model...");
    let mut model = OutsideInsideModel::new();
    log::info!("Model initialized.");
    server
        .fn_handler("/camera", Method::Get, move |request| {
            match pt_http.try_lock() {
                Ok(mut pt_unlocked) => {
                    pt_unlocked[0] = 1;
                    let mut response = request.into_response(200, Some("OK"), &[]).unwrap();
                    response.write("NOT done".as_bytes())?;
                }
                Err(_) => {
                    let mut response = request.into_ok_response()?;
                    response.write("no framebuffer".as_bytes())?;
                }
            }

            Ok::<(), EspIOError>(())
        })
        .unwrap();

    log::info!("Opening dataset directories...");
    let images_dir = volume_mgr.open_dir(root_dir, "photos").unwrap();
    let label_0_dir = volume_mgr.open_dir(root_dir, "ph_lab").unwrap();
    let mut labels = vec![];

    // save_bulk_images_and_take_commands(
    //     &camera,
    //     &volume_mgr,
    //     label_0_dir,
    //     mutex_loop.clone(),
    //     command_loop.clone()
    // );
    volume_mgr
        .iterate_dir(label_0_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels.push(entry.name.to_string());
            }
        })
        .ok();
    log::info!("found {} label 0 images", labels.len());
    // save_bulk_images_and_take_commands(
    //     &camera,
    //     &volume_mgr,
    //     images_dir,
    //     mutex_loop.clone(),
    //     command_loop.clone()
    // );
    volume_mgr
        .iterate_dir(images_dir, |entry| {
            if !entry.attributes.is_directory() {
                labels.push(entry.name.to_string());
            }
        })
        .ok();
    log::info!("found {} label 1 images", labels.len());
    // panic!("Starting training...");
    log::info!("\r\nstop\r\n");
    // volume_mgr.close_dir(label_0_dir).unwrap();
    // volume_mgr.close_dir(images_dir).unwrap();
    // let test_0 = volume_mgr.open_dir(root_dir, "test_0").unwrap();
    // let test_1 = volume_mgr.open_dir(root_dir, "test_1").unwrap();
    // validation_loop(&volume_mgr, test_0, test_1, image_mutex.clone(), &mut model);
    // volume_mgr.close_dir(test_0).unwrap();
    // volume_mgr.close_dir(test_1).unwrap();
    // let images_dir = volume_mgr.open_dir(root_dir, "photos").unwrap();
    // let label_0_dir = volume_mgr.open_dir(root_dir, "ph_lab").unwrap();
    training_loop(&volume_mgr, label_0_dir, images_dir, image_mutex.clone(), &mut model, EPOCHS);
    volume_mgr.close_dir(label_0_dir).unwrap();
    volume_mgr.close_dir(images_dir).unwrap();
    // save_bulk_images_and_take_commands(
    //     &camera,
    //     &volume_mgr,
    //     test_0,
    //     mutex_loop.clone(),
    //     command_loop.clone()
    // );
    // save_bulk_images_and_take_commands(
    //     &camera,
    //     &volume_mgr,
    //     test_1,
    //     mutex_loop.clone(),
    //     command_loop.clone()
    // );
    let test_0 = volume_mgr.open_dir(root_dir, "test_0").unwrap();
    let test_1 = volume_mgr.open_dir(root_dir, "test_1").unwrap();
    validation_loop(&volume_mgr, test_0, test_1, image_mutex.clone(), &mut model);
    panic!();
    let response_mutex = Arc::new(Mutex::new([0u8; 1]));
    let response_mutex_http = response_mutex.clone();
    server
        .fn_handler("/camera_with_prediction.raw", Method::Get, move |request| {
            match mutex_http.try_lock() {
                Ok(image_unlocked) => {
                    let body: &[u8] = cast_slice(image_unlocked.as_slice());
                    match response_mutex_http.try_lock() {
                        Ok(unlocked_response) => {
                            let mut response = request
                                .into_response(
                                    200,
                                    Some("OK"),
                                    &[("prediction", format!("{}", unlocked_response[0]).as_str())]
                                )
                                .unwrap();
                            response.write(body)?;
                        }
                        Err(_) => {
                            log::info!("no response mutex");
                        }
                    }
                }
                Err(_) => {
                    let mut response = request.into_ok_response()?;
                    response.write("no framebuffer".as_bytes())?;
                }
            }

            Ok::<(), EspIOError>(())
        })
        .unwrap();
    loop {
        unsafe {
            esp_idf_sys::vTaskDelay(100);
        }
        log::info!("Getting framebuffer...");
        let framebuffer = camera.get_framebuffer();
        log::info!("Framebuffer obtained.");
        log::info!("sampling image...");
        if let Some(framebuffer) = framebuffer {
            let data = framebuffer.data();
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

                    log::info!("predicting");
                    let prediction = model.predict([**unlocked_image]);
                    log::info!("prediction: {}, {}", prediction[0], prediction[1]);
                    if let Ok(mut response) = response_mutex.try_lock() {
                        response[0] = if prediction[0] > prediction[1] {
                            0
                        } else if prediction[1] > prediction[0] {
                            1
                        } else {
                            2
                        };
                    }
                }
                Err(_) => {
                    log::info!("no mutex");
                }
            }
        } else {
            log::info!("no framebuffer");
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
                        6 => "start_readings",
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
}

fn downsample_rgb565(
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
fn read_rgb565_pixel(data: &[u8], pixel_index: usize) -> u16 {
    let byte_index = pixel_index * 2;

    // Read as little-endian: low byte first, then high byte
    let low = data[byte_index] as u16;
    let high = data[byte_index + 1] as u16;

    // Combine: high byte in upper 8 bits, low byte in lower 8 bits
    (high << 8) | low
}

/// Alternative: Read as big-endian if the above doesn't work
#[inline]
fn read_rgb565_pixel_be(data: &[u8], pixel_index: usize) -> u16 {
    let byte_index = pixel_index * 2;

    // Read as big-endian: high byte first, then low byte
    let high = data[byte_index] as u16;
    let low = data[byte_index + 1] as u16;

    (high << 8) | low
}

fn bilinear_rgb565_f32(p00: u16, p10: u16, p01: u16, p11: u16, fx: f32, fy: f32) -> [f32; 3] {
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
fn test_rgb565_byte_order(data: &[u8]) {
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
fn downsample_rgb565_be(
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

fn save_image<'a, R: BlockDevice, S: TimeSource>(
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

fn open_image<'a, R: BlockDevice, S: TimeSource>(
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
fn verify_image_saved<'a, R: BlockDevice, S: TimeSource>(
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

fn save_bulk_images_and_take_commands(
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
fn training_loop(
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
            print_stack_pointer();
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
                print_stack_pointer();
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

fn validation_loop(
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

fn print_stack_pointer() {
    // Prefer a real inline-asm read of the SP when the optional feature is enabled
    // and the target architecture is supported. Otherwise fall back to the
    // address-of-local workaround which approximates the current stack pointer.
    //
    // To enable the inline asm path, build with:
    //   cargo build --features use-inline-asm
    //
    // Note: inline asm requires a toolchain that supports it for your target.
    #[cfg(all(feature = "use-inline-asm"))]
    {
        let sp: usize;
        unsafe {
            core::arch::asm!("mov {0}, a1", out(reg) sp);
        }
        log::info!("Stack Pointer : {:#X}", sp);
        return;
    }
}