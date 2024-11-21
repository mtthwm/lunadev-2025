use gputter::build_shader;

// Generates a list of gradient magnitudes from a given heightmap
build_shader!(
    pub(crate) Height2Grad,
    r#"
    const HEIGHTMAP_WIDTH: NonZeroU32 = {{heightmap_width}};
    const CELL_SIZE: f32 = {{cell_size}};
    const CELL_COUNT: NonZeroU32 = {{cell_count}};

    const INF = 3e38;

    #[buffer(HostWriteOnly)] var<storage, read> original_heightmap: array<f32, CELL_COUNT>;
    #[buffer(HostReadOnly)] var<storage, read_write> gradient_map: array<f32, CELL_COUNT>;

    fn coord_to_index (x: u32, y: u32) -> u32  {
        return (HEIGHTMAP_WIDTH)*y + x;
    }

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        var ix = global_id.x;
        var iy = global_id.y;
        let index = coord_to_index(ix, iy);

        if (ix == HEIGHTMAP_WIDTH-1 || iy == HEIGHTMAP_WIDTH-1) {
            gradient_map[index] = INF;
        } else {
            let eastIndex = coord_to_index(ix + 1, iy);
            let southIndex = coord_to_index(ix, iy + 1);

            // TODO: IMPROVE THIS TERMINOLOGY (-Z should be forward, but this function assumes z is up)

            let dzdx = original_heightmap[eastIndex] - original_heightmap[index];
            let dzdy = original_heightmap[southIndex] - original_heightmap[index];
            
            let gradx = dzdx / CELL_SIZE;
            let grady = dzdy / CELL_SIZE;
            let gradMag = sqrt(gradx*gradx + grady*grady);

            gradient_map[index] = gradMag;
        }
    }
    "#
);
