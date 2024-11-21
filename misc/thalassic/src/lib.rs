use std::num::NonZeroU32;

use bytemuck::cast_slice_mut;
use depth2pcl::Depth2Pcl;
use gputter::{
    buffers::{
        storage::{HostReadOnly, HostWriteOnly, ShaderReadOnly, ShaderReadWrite, StorageBuffer},
        uniform::UniformBuffer,
        GpuBufferSet,
    },
    compute::ComputePipeline,
    shader::BufferGroupBinding,
    types::{AlignedMatrix4, AlignedVec4},
};
use nalgebra::{Vector2, Vector3};
use pcl2height::Pcl2Height;
use height2gradient::Height2Grad;

pub mod depth2pcl;
pub mod pcl2height;
pub mod height2gradient;
mod clustering;
pub use clustering::Clusterer;

/// 1. Depths
/// 2. Transform
type DepthBindGrp = (
    StorageBuffer<[u32], HostWriteOnly, ShaderReadOnly>,
    UniformBuffer<AlignedMatrix4<f32>>,
);

/// Points used by depth2pcl and pcl2height
type PointsBindGrp = (StorageBuffer<[AlignedVec4<f32>], HostReadOnly, ShaderReadWrite>,);

/// Heightmap used by pcl2height and height2grad
type HeightMapBindGrp = (StorageBuffer<[u32], HostReadOnly, ShaderReadWrite>,);

/// Original heightmap used by pcl2height
type PclBindGrp = (StorageBuffer<[f32], HostWriteOnly, ShaderReadOnly>,);

type BindGroups = (
    GpuBufferSet<DepthBindGrp>,
    GpuBufferSet<PointsBindGrp>,
    GpuBufferSet<HeightMapBindGrp>,
    GpuBufferSet<PclBindGrp>,
);

#[derive(Debug, Clone, Copy)]
pub struct ThalassicBuilder {
    pub image_width: NonZeroU32,
    pub focal_length_px: f32,
    pub principal_point_px: Vector2<f32>,
    pub depth_scale: f32,
    pub pixel_count: NonZeroU32,
    pub heightmap_width: NonZeroU32,
    pub cell_size: f32,
    pub cell_count: NonZeroU32,
}

impl ThalassicBuilder {
    pub fn build(self) -> ThalassicPipeline {
        let bind_grps = (
            GpuBufferSet::from((
                StorageBuffer::new_dyn(self.pixel_count.get() as usize).unwrap(),
                UniformBuffer::new(),
            )),
            GpuBufferSet::from((StorageBuffer::new_dyn(self.pixel_count.get() as usize).unwrap(),)),
            GpuBufferSet::from((StorageBuffer::new_dyn(self.cell_count.get() as usize).unwrap(),)),
            GpuBufferSet::from((StorageBuffer::new_dyn(self.cell_count.get() as usize).unwrap(),)),
        );

        let [depth_fn] = Depth2Pcl {
            depths: BufferGroupBinding::<_, BindGroups>::get::<0, 0>(),
            points: BufferGroupBinding::<_, BindGroups>::get::<1, 0>(),
            transform: BufferGroupBinding::<_, BindGroups>::get::<0, 1>(),
            image_width: self.image_width,
            focal_length_px: self.focal_length_px,
            principal_point_px: self.principal_point_px.into(),
            depth_scale: self.depth_scale,
            pixel_count: self.pixel_count,
        }
        .compile();

        let [height_fn] = Pcl2Height {
            points: BufferGroupBinding::<_, BindGroups>::get::<1, 0>(),
            heightmap: BufferGroupBinding::<_, BindGroups>::get::<2, 0>(),
            cell_size: self.cell_size,
            heightmap_width: self.heightmap_width,
            cell_count: self.cell_count,
            original_heightmap: BufferGroupBinding::<_, BindGroups>::get::<3, 0>(),
            projection_width: self.image_width,
            point_count: self.pixel_count,
        }
        .compile();

        let mut pipeline = ComputePipeline::new([&depth_fn, &height_fn]);
        pipeline.workgroups = [
            Vector3::new(
                self.image_width.get(),
                self.pixel_count.get() / self.image_width,
                1,
            ),
            Vector3::new(
                self.heightmap_width.get(),
                self.cell_count.get() / self.heightmap_width,
                2 * (self.image_width.get() - 1) * (self.pixel_count.get() / self.image_width - 1),
            ),
        ];
        ThalassicPipeline {
            pipeline,
            bind_grps,
        }
    }
}

pub struct ThalassicPipeline {
    pipeline: ComputePipeline<BindGroups, 2>,
    bind_grps: BindGroups,
}

impl ThalassicPipeline {
    pub fn provide_depths(
        &mut self,
        depths: &[u32],
        camera_transform: &AlignedMatrix4<f32>,
        out_pcl: &mut [AlignedVec4<f32>],
        out_heightmap: &mut [f32],
    ) {
        self.pipeline
            .new_pass(|mut lock| {
                self.bind_grps.0.write::<0, _>(depths, &mut lock);
                self.bind_grps.0.write::<1, _>(camera_transform, &mut lock);
                self.bind_grps
                    .2
                    .buffers
                    .0
                    .copy_into(&mut self.bind_grps.3.buffers.0, &mut lock);
                &mut self.bind_grps
            })
            .finish();
        self.bind_grps.1.buffers.0.read(out_pcl);
        self.bind_grps
            .2
            .buffers
            .0
            .read(cast_slice_mut(out_heightmap));
    }
}


#[cfg(test)]
mod test_height2gradient {
    use gputter::init_gputter_blocking;

    use super::*;

    struct GradMapTestPipeline {
        pipeline: ComputePipeline<GradTestBindGroups, 1>,
        bind_grps: GradTestBindGroups,
    }

    type HeightBindGrp = (StorageBuffer<[f32], HostWriteOnly, ShaderReadOnly>,);
    type GradBindGrp = (StorageBuffer<[f32], HostReadOnly, ShaderReadWrite>,);

    type GradTestBindGroups = (
        GpuBufferSet<HeightBindGrp>,
        GpuBufferSet<GradBindGrp>,
    );

    impl GradMapTestPipeline {
        pub fn build(cell_count: u32, width: u32) -> GradMapTestPipeline {
            let height_buff = StorageBuffer::new_dyn(cell_count as usize).unwrap();
            let grad_buff = StorageBuffer::new_dyn(cell_count as usize).unwrap();
            let bind_grps: GradTestBindGroups = (
                GpuBufferSet::from((height_buff,)),
                GpuBufferSet::from((grad_buff,)),
            );

            let [grad_fn] = Height2Grad {
                original_heightmap: BufferGroupBinding::<StorageBuffer<[f32], _, _>, GradTestBindGroups>::get::<0, 0>(),
                gradient_map: BufferGroupBinding::<StorageBuffer<[f32], _, _>, GradTestBindGroups>::get::<1, 0>(),
                cell_count: NonZeroU32::new(cell_count).unwrap(),
                heightmap_width: NonZeroU32::new(width).unwrap(),
                cell_size: 1.0,
            }
            .compile();

            let mut pipeline = ComputePipeline::new([&grad_fn]);
            pipeline.workgroups = [
                Vector3::new(
                    width as u32,
                    (cell_count / width) as u32,
                    1,
                )
            ];
            GradMapTestPipeline {
                pipeline,
                bind_grps,
            }  
        }
    }

    impl GradMapTestPipeline {
        fn provide_heights (
            &mut self,
            heights: &[f32],
            gradients: &mut[f32],
        ) {
            self.pipeline
                .new_pass(|mut lock| {
                    self.bind_grps.0.write::<0, _>(heights, &mut lock);
                    &mut self.bind_grps
                })
                .finish();
            self.bind_grps.1.buffers.0.read(gradients);
        }
    }

    #[test]
    fn basic_grad_map() {
        const INF: f32 = 3.0e38;
        let mut original: [f32; 9] = [
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0
        ];
        let expected: [f32; 9] = [
            1.0, 1.0, INF,
            1.0, 1.0, INF,
            INF, INF, INF
        ];
        let mut gradients: [f32; 9] = [0.0; 9];
        let _ = init_gputter_blocking();
        let mut pipeline = GradMapTestPipeline::build(9, 3);
        pipeline.provide_heights(&mut original, &mut gradients);

        assert_eq!(gradients, expected);
    }

    #[test]
    fn diagonal_grad_map() {
        const INF: f32 = 3.0e38;
        const RT2: f32 = 1.41421356237;

        let mut original: [f32; 16] = [
            0.0, 1.0, 2.0, 3.0,
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 5.0,
            3.0, 4.0, 5.0, 6.0
        ];
        let expected: [f32; 16] = [
            RT2, RT2, RT2, INF,
            RT2, RT2, RT2, INF,
            RT2, RT2, RT2, INF,
            INF, INF, INF, INF
        ];
        let mut gradients: [f32; 16] = [0.0; 16];
        let _ = init_gputter_blocking();
        let mut pipeline = GradMapTestPipeline::build(16, 4);
        pipeline.provide_heights(&mut original, &mut gradients);

        assert_eq!(gradients, expected);
    }
}