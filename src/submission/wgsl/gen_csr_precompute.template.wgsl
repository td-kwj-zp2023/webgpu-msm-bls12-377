@group(0) @binding(0)
var<storage, read> scalar_chunks: array<u32>;
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

    // TODO: implement precomputation
    new_point_indices[id] = id;
    cluster_start_indices[id] = id;
    cluster_end_indices[id] = id + 1u;
}
