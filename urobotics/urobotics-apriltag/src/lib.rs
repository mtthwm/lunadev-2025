//! This crate provides a node that can identify apriltags
//! in images.

use std::{f64::consts::PI, fmt::Debug};

use apriltag::{families::TagStandard41h12, DetectorBuilder, Image, TagParams};
use apriltag_image::{image::ImageBuffer, ImageExt};
use apriltag_nalgebra::PoseExt;
use fxhash::FxHashMap;
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
use urobotics_core::{
    define_callbacks, fn_alias,
    log::{error, warn},
    shared::SharedDataReceiver,
};

pub use apriltag_image::image;

define_callbacks!(DetectionCallbacks => Fn(detection: TagObservation) + Send + Sync);
fn_alias! {
    pub type DetectionCallbacksRef = CallbacksRef(TagObservation) + Send + Sync
}

/// An observation of the global orientation and position
/// of the camera that observed an apriltag.
#[derive(Clone, Copy)]
pub struct TagObservation {
    /// The orientation and position of the apriltag relative to the observer.
    pub tag_local_isometry: Isometry3<f64>,
    /// The orientation and position of the apriltag in global space.
    ///
    /// These are the same values that were passed to `add_tag`. As such,
    /// if these values were not known then, this value will be incorrect.
    /// However, this can be set to the correct value, allowing
    /// `get_isometry_of_observer` to produce correct results.
    pub tag_global_isometry: Isometry3<f64>,
    /// The goodness of an observation.
    ///
    /// This is a value generated by the apriltag detector.
    pub decision_margin: f32,
}

impl Debug for TagObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseObservation")
            .field("pose", &self.tag_local_isometry)
            .field("decision_margin", &self.decision_margin)
            .finish()
    }
}

impl TagObservation {
    /// Get the isometry of the observer.
    pub fn get_isometry_of_observer(&self) -> Isometry3<f64> {
        // let mut observer_pose = self.tag_local_isometry;
        // observer_pose.translation.vector = self.tag_global_isometry.translation.vector
        //     + self.tag_global_isometry.rotation
        //         * observer_pose.rotation.inverse()
        //         * observer_pose.translation.vector;
        // observer_pose.rotation = self.tag_global_isometry.rotation
        //     * UnitQuaternion::from_axis_angle(&(observer_pose.rotation * Vector3::y_axis()), PI)
        //     * observer_pose.rotation;
        // observer_pose
        self.tag_global_isometry * self.tag_local_isometry.inverse()
    }
}

struct KnownTag {
    pose: Isometry3<f64>,
    tag_params: TagParams,
}

/// A Node that can detect apriltags in images.
///
/// Actual detection does not occur until the node
/// is running.
pub struct AprilTagDetector {
    img_subscriber: SharedDataReceiver<ImageBuffer<image::Luma<u8>, Vec<u8>>>,
    detection_callbacks: DetectionCallbacks,
    known_tags: FxHashMap<usize, KnownTag>,
    pub focal_length_x_px: f64,
    pub focal_length_y_px: f64,
    pub image_width: u32,
    pub image_height: u32,
}

impl AprilTagDetector {
    /// Creates a new detector for a specific camera.
    ///
    /// The given `image_sub` must produce images that
    /// have a width of `image_width` and a height of
    /// `image_height`, and the camera that produced it
    /// must have a focal length, in pixels, of `focal_length_px`.
    ///
    /// As such, it is strongly encouraged that the subscription
    /// should not be a sum of multiple subscriptions.
    pub fn new(
        focal_length_x_px: f64,
        focal_length_y_px: f64,
        image_width: u32,
        image_height: u32,
        img_subscriber: SharedDataReceiver<ImageBuffer<image::Luma<u8>, Vec<u8>>>,
    ) -> Self {
        Self {
            img_subscriber,
            detection_callbacks: DetectionCallbacks::default(),
            known_tags: Default::default(),
            focal_length_x_px,
            focal_length_y_px,
            image_width,
            image_height,
        }
    }

    /// Add a 41h12 tag to look out for. All units are in meters.
    ///
    /// Orientations and positions should be in global space. If this
    /// is not known, any value can be used. However, [`TagObservation::get_isometry_of_observer`]
    /// will not produce correct results in that case.
    pub fn add_tag(
        &mut self,
        tag_position: Point3<f64>,
        tag_orientation: UnitQuaternion<f64>,
        tag_width: f64,
        tag_id: usize,
    ) {
        self.known_tags.insert(
            tag_id,
            KnownTag {
                pose: Isometry3::from_parts(tag_position.into(), tag_orientation),
                tag_params: TagParams {
                    tagsize: tag_width,
                    fx: self.focal_length_x_px,
                    fy: self.focal_length_y_px,
                    cx: self.image_width as f64 / 2.0,
                    cy: self.image_height as f64 / 2.0,
                },
            },
        );
    }

    pub fn detection_callbacks_ref(&self) -> DetectionCallbacksRef {
        self.detection_callbacks.get_ref()
    }
}

impl AprilTagDetector {
    pub fn run(mut self) {
        let mut detector = DetectorBuilder::new()
            .add_family_bits(TagStandard41h12::default(), 1)
            .build()
            .unwrap();

        loop {
            let img = self.img_subscriber.get();
            if img.width() != self.image_width || img.height() != self.image_height {
                error!(
                    "Received incorrectly sized image: {}x{}",
                    img.width(),
                    img.height()
                );
                continue;
            }
            let img = Image::from_image_buffer(&img);

            for detection in detector.detect(&img) {
                if detection.decision_margin() < 130.0 {
                    continue;
                }
                let Some(known) = self.known_tags.get(&detection.id()) else {
                    continue;
                };
                let Some(tag_local_isometry) = detection.estimate_tag_pose(&known.tag_params)
                else {
                    warn!("Failed to estimate pose of {}", detection.id());
                    continue;
                };
                let mut tag_local_isometry = tag_local_isometry.to_na();
                tag_local_isometry.translation.y *= -1.0;
                tag_local_isometry.translation.z *= -1.0;
                let mut scaled_axis = tag_local_isometry.rotation.scaled_axis();
                scaled_axis.y *= -1.0;
                scaled_axis.z *= -1.0;
                tag_local_isometry.rotation = UnitQuaternion::from_scaled_axis(scaled_axis);
                tag_local_isometry.rotation = UnitQuaternion::from_scaled_axis(tag_local_isometry.rotation * Vector3::new(0.0, PI, 0.0)) * tag_local_isometry.rotation;

                self.detection_callbacks.call(TagObservation {
                    tag_local_isometry,
                    decision_margin: detection.decision_margin(),
                    tag_global_isometry: known.pose,
                });
            }
        }
    }
}
