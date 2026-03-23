# ESP32-CAM On-Device CNN Fine-Tuning

Experiment demonstrating on-device learning with a CNN running on an ESP32-CAM.  
The device collects image data and saves to sd card, fine-tunes the model locally, and evaluates on newly captured samples.

## Features
- Image capture via ESP32-CAM
- Embedded CNN inference
- On-device fine-tuning
- Real-time evaluation with new data

## Requirements
- ESP32-CAM
- Rust (with `xtensa-esp32-espidf` target)
- `espflash`

## Build & Flash

```bash
cargo build --release
espflash flash --monitor ./target/xtensa-esp32-espidf/release/esp32_cam_microflow
