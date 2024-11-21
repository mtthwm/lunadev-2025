use std::os::unix::ffi::OsStrExt;

use anyhow::Context;
use fxhash::{FxHashMap, FxHashSet};
use udev::Udev;
use urobotics::{
    log::{error, warn},
    shared::{OwnedData, SharedData},
};
use urobotics_apriltag::{
    image::{self, ImageBuffer, Luma},
    AprilTagDetector,
};
use v4l::{buffer::Type, format, io::traits::CaptureStream, prelude::MmapStream, video::Capture};

use crate::localization::LocalizerRef;

pub struct CameraInfo {
    pub k_node: k::Node<f64>,
    pub focal_length_x_px: f64,
    pub focal_length_y_px: f64,
}

pub fn enumerate_cameras(
    localizer_ref: LocalizerRef,
    port_to_chain: impl IntoIterator<Item = (String, CameraInfo)>,
) -> anyhow::Result<()> {
    let mut port_to_chain: FxHashMap<String, Option<_>> = port_to_chain
        .into_iter()
        .map(|(port, chain)| (port, Some(chain)))
        .collect();
    {
        let udev = Udev::new()?;
        let mut enumerator = udev::Enumerator::with_udev(udev.clone())?;
        let mut seen = FxHashSet::default();

        for udev_device in enumerator.scan_devices()? {
            let Some(path) = udev_device.devnode() else { continue; };
            // Valid camera paths are of the form /dev/videoN
            let Some(path_str) = path.to_str() else { continue; };
            if !path_str.starts_with("/dev/video") {
                continue;
            }
            if let Some(name) = udev_device.attribute_value("name") {
                if let Some(name) = name.to_str() {
                    if name.contains("RealSense") {
                        continue;
                    }
                }
            }
            let Some(port_raw) = udev_device.property_value("ID_PATH") else {
                warn!("No port for camera {path_str}");
                continue;
            };
            let Some(port) = port_raw.to_str() else {
                warn!("Failed to parse port of camera {path_str}");
                continue;
            };
            if !seen.insert(port.to_string()) {
                continue;
            }
            let Some(cam_info) = port_to_chain.get_mut(port) else {
                warn!("Unexpected camera with port {}", port);
                continue;
            };
            let CameraInfo {
                k_node,
                focal_length_x_px,
                focal_length_y_px
            } = cam_info.take().unwrap();

            let mut camera = match v4l::Device::with_path(path) {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to open camera {path_str}: {e}");
                    continue;
                }
            };

            let format = match camera.format() {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to get format for camera {path_str}: {e}");
                    continue;
                }
            };
            let image = OwnedData::from(ImageBuffer::from_pixel(
                format.width,
                format.height,
                Luma([0]),
            ));
            let mut image = image.pessimistic_share();
            let det = AprilTagDetector::new(
                focal_length_x_px,
                focal_length_y_px,
                format.width,
                format.height,
                image.create_lendee(),
            );
            let localizer_ref = localizer_ref.clone();
            let mut local_transform = k_node.origin();
            local_transform.inverse_mut();
            det.detection_callbacks_ref().add_fn(move |observation| {
                localizer_ref.set_april_tag_isometry(
                    local_transform * observation.get_isometry_of_observer(),
                );
            });
            std::thread::spawn(move || det.run());

            std::thread::spawn(move || {
                let mut stream = MmapStream::with_buffers(&mut camera, Type::VideoCapture, 4)
                    .expect("Failed to create buffer stream");

                loop {
                    let (buf, _) = stream.next().unwrap();
                    match image.try_recall() {
                        Ok(mut img) => {
                            img.copy_from_slice(buf);
                            image = img.pessimistic_share();
                        }
                        Err(img) => {
                            image = img;
                        }
                    }
                }
            });
        }
    }

    for (port, cam_info) in port_to_chain {
        if cam_info.is_some() {
            error!("Camera with port {port} not found");
        }
    }

    Ok(())
}
