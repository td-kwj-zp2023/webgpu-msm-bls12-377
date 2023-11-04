{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> points: array<Point>;
@group(0) @binding(1)
var<storage, read> scalar_chunks: array<u32>;
@group(0) @binding(2)
var<storage, read> new_point_indices: array<u32>;
@group(0) @binding(3)
var<storage, read> cluster_start_indices: array<u32>;
@group(0) @binding(4)
var<storage, read_write> new_points: array<Point>;
@group(0) @binding(5)
var<storage, read_write> new_scalar_chunks: array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    /*new_points[global_id.x] = points[global_id.x];*/

    var gidx = global_id.x;
    var gidy = global_id.y;

    var start_idx = cluster_start_indices[gidx];
    var end_idx = cluster_start_indices[gidx + 1u];

    var pt = points[gidy + new_point_indices[gidy + start_idx]];

    for (var i = start_idx + 1u; i < end_idx; i ++) {
          pt = add_points(pt, points[gidy + new_point_indices[gidy + i]]);
    }

    new_points[gidx] = pt;
    new_scalar_chunks[gidx] = scalar_chunks[gidy + new_point_indices[gidy + start_idx]];
}
