// Input
@group(0) @binding(0)
var<storage, read> scalar_chunks: array<u32>;

// Output
@group(0) @binding(1)
var<storage, read_write> new_point_indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> cluster_start_indices: array<u32>;
@group(0) @binding(3)
var<storage, read_write> cluster_end_indices: array<u32>;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    // Each thread handles a row, so the number of rows is the number of
    // threads
    let num_rows = {{ num_rows }}u;

    // Max cluster size
    let m = {{ max_cluster_size }}u;

    // Initialise the 2D array
    var map: array<array<u32, {{ max_cluster_size}}>, {{ max_chunk_val }}>;

    // The keys to the map
    var keys: array<u32, {{ max_chunk_val }}>;

    // The number of keys
    var num_keys = 0u;

    // The overflow array
    var overflow: array<vec2<u32>, {{ overflow_size }}>;
    var num_overflow = 0u;

    let row_size = {{ row_size }}u;

    // For each scalar chunk in the row
    for (var i = 0u; i < row_size; i ++) {
        let pt_idx = id * row_size + i;
        let chunk = scalar_chunks[pt_idx];

        // Ignore 0s
        if (chunk == 0u) {
            continue;
        }

        // If this chunk has not been seen, store it in keys
        if (map[chunk][0] == 0u) {
            keys[num_keys] = chunk;
            num_keys ++;
        }

        // If the chunk has been seen m times, place it in an overflow array
        if (map[chunk][0] > m) {
            overflow[num_overflow] = vec2(chunk, pt_idx);
            num_overflow ++;
		}

        let cluster_current_size = map[chunk][0];
        map[chunk][cluster_current_size + 1u] = pt_idx;
        map[chunk][0] = map[chunk][0] + 1u;
    }

    var k = 0u;
    for (var i = 0u; i < num_keys; i ++) {
        let chunk = keys[i];
        let start = k;
        for (var j = 0u; j < map[chunk][0]; j ++) {
            new_point_indices[id * row_size + k] = map[chunk][j];
            k ++;
        }
        cluster_start_indices[id * row_size + i] = start;
        cluster_end_indices[id * row_size + i] = start + map[chunk][0];
    }

    // TODO: handle overflow
    for (var i = 0u; i < num_overflow; i ++) {
        let chunk = overflow[i][0];
        let pt_idx = overflow[i][1];
        // TODO: fix the following
        new_point_indices[id * row_size + k + i] = pt_idx;
        cluster_start_indices[id * row_size + num_keys + i] = start;
        cluster_end_indices[id * row_size + num_keys + i] = start + 1u;
    }
}
