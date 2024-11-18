use gputter::build_shader;

build_shader!(
    pub(crate) Height2Grad,
    r#"
    const HEIGHTMAP_WIDTH: NonZeroU32 = {{heightmap_width}};
    const CELL_SIZE: f32 = {{cell_size}};
    const CELL_COUNT: NonZeroU32 = {{cell_count}};

    #[buffer(HostWriteOnly)] var<storage, read> original_heightmap: array<f32, CELL_COUNT>;
    #[buffer(HostReadOnly)] var<storage, read_write> gradient_map: array<f32, CELL_COUNT>;

    fn coord_to_index (x: u32, y: u32) -> u32  {
        return HEIGHTMAP_WIDTH*y + x;
    }

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x = global_id.x;
        let y = global_id.y;
        let index = coord_to_index(x, y);
        gradient_map[index] = original_heightmap[index] + 1;
    }
    "#
);
