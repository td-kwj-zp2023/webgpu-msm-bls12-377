// Input
@group(0) @binding(0)
var<storage, read> scalar_chunks: array<u32>;
@group(0) @binding(1)
var<storage, read> subtask_idx: u32;

// Output
@group(0) @binding(2)
var<storage, read_write> new_point_indices: array<u32>;
@group(0) @binding(3)
var<storage, read_write> cluster_start_indices: array<u32>;
@group(0) @binding(4)
var<storage, read_write> cluster_end_indices: array<u32>;

// Intermediate buffers
@group(0) @binding(5)
var<storage, read_write> map: array<array<u32, {{ max_cluster_size_plus_one }}>, {{ max_chunk_val }}>;
@group(0) @binding(6)
var<storage, read_write> overflow: array<u32, {{ overflow_size }}>;
@group(0) @binding(7)
var<storage, read_write> keys: array<u32, {{ max_chunk_val }}>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Max cluster size
    let max_cluster_size = {{ max_cluster_size }}u; 

    // The number of keys
    var num_keys = 0u;

    // The overflow array size
    var num_overflow = 0u;

    // The number of nth chunks across all the scalars
    let num_chunks = {{ num_chunks }}u;

    // Number of subtasks
    let num_subtasks = {{ num_subtasks }}u;

    // Populate the cluster map, which is a 2D array. Each array within the map
    // has the following structure of n + 1 values:
    // [n, item_0, ..., item_n]
    // The 0th value (n) is the cluster size.
    // Also populate the keys array, which contains the keys of the map.
    for (var i = 0u; i < num_chunks; i++) {
        let pt_idx = subtask_idx * num_chunks + i;
        let chunk = scalar_chunks[pt_idx];

        // Ignore 0s
        if (chunk == 0u) {
            continue;
        }

        let cluster_size = map[chunk][0];

        // If this chunk has not been seen, update the keys array
        if (cluster_size == 0u) {
            keys[num_keys] = chunk;
            num_keys ++;
        }

        // If the cluster is full, place pt_idx in overflow
        if (cluster_size >= max_cluster_size) {
            overflow[num_overflow] = pt_idx;
            num_overflow ++;
		} else {
            // Otherwise, store pt_idx in the map
            map[chunk][cluster_size + 1u] = pt_idx;
            map[chunk][0] = cluster_size + 1u;
        }
    }

    // Populate new_point_indices, cluster_start_indices and
    // cluster_end_indices

    // cluster_idx tracks the values set in cluster_start_indices and
    // cluster_end_indices as we iterate over the keys
    var cluster_idx = 0u;
    var k = 0u;
    for (var i = 0u; i < num_keys; i ++) {
        let chunk = keys[i];
        let cluster_size = map[chunk][0];

        // Rearrange the point indices
        for (var j = 1u; j < cluster_size + 1u; j ++) {
            new_point_indices[k] = map[chunk][j];
            k ++;
        }

        // Set cluster indices
        cluster_start_indices[i] = cluster_idx;
        cluster_idx += cluster_size;
        cluster_end_indices[i] = cluster_idx;
    }

    // Handle overflow
    for (var i = 0u; i < num_overflow; i ++) {
        let pt_idx = overflow[i];
        new_point_indices[k] = pt_idx;
        k ++;

        let cidx = num_keys + i;
        cluster_start_indices[cidx] = cluster_idx;
        cluster_idx ++;
        cluster_end_indices[cidx] = cluster_idx;
    }

    /*// For each scalar chunk in the row*/
    /*for (var i = 0u; i < row_size; i++) {*/
        /*let pt_idx = subtask_idx * row_size + i;*/
        /*let chunk = scalar_chunks[pt_idx];*/

        /*// Ignore 0s*/
        /*if (chunk == 0u) {*/
            /*continue;*/
        /*}*/

        /*// If this chunk has not been seen, store it in keys*/
        /*if (map[chunk][0] == 0u) {*/
            /*keys[num_keys] = chunk;*/
            /*num_keys ++;*/
        /*}*/

        /*// If the chunk has been seen at least max_cluster_size times, place it in the*/
        /*// overflow array*/
        /*let cluster_current_size = map[chunk][0];*/
        /*if (cluster_current_size + 1u > max_cluster_size - 1u) {*/
            /*overflow[num_overflow] = pt_idx;*/
            /*overflow_size[num_overflow] = (cluster_current_size + 1u) - (max_cluster_size - 1u);*/
            /*num_overflow ++;*/
		/*} else {*/
            /*// Otherwise, store it in the map*/
            /*map[chunk][cluster_current_size + 1u] = pt_idx;*/
            /*map[chunk][0] = cluster_current_size + 1u;*/
        /*}*/
    /*}*/

    /*var k = 0u;*/
    /*for (var i = 0u; i < num_keys; i++) {*/
        /*let chunk = keys[i];*/
        /*let chunk_len = map[chunk][0];*/
        /*let start = k;*/
        /*for (var j = 1u; j < chunk_len; j ++) {*/
            /*new_point_indices[subtask_idx * row_size + k] = map[chunk][j];*/
            /*k ++;*/
        /*}*/
        /*cluster_start_indices[subtask_idx * row_size + i] = start;*/
        /*cluster_end_indices[subtask_idx * row_size + i] = start + chunk_len;*/
    /*}*/
    
    /*var last_cluster_start_index = cluster_start_indices[subtask_idx * row_size + num_keys - 1];*/

    /*// Handle overflow*/
    /*for (var i = 0u; i < num_overflow; i ++) {*/
        /*var pt_idx = overflow[i];*/
        /*new_point_indices[subtask_idx * row_size + k + i] = pt_idx;*/

        /*last_cluster_start_index ++;*/

        /*cluster_start_indices[subtask_idx * row_size + num_keys + i] = */
            /*last_cluster_start_index + overflow_size[i] + 1;*/

        /*cluster_end_indices[subtask_idx * row_size + num_keys + i] =*/
            /*last_cluster_start_index + overflow_size[i] + 1;*/
    /*}*/
}
