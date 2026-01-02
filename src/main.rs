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
use esp32_cam_microflow::read_image::downsample_rgb565;
use esp32_cam_microflow::read_image::open_image;
use esp32_cam_microflow::training::OutsideInsideModel;
use esp32_cam_microflow::training::training_loop;
use esp32_cam_microflow::training::validation_loop;
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
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f32 = 0.01;
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
    // #[cfg(feature = "use-inline-asm")]
    // {
    //     print_stack_pointer();
    // }
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
