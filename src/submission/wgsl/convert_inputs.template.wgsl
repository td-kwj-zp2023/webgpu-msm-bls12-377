{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<BigInt>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    var pt = points[gidx];
    var scalar = scalars[gidx];

    // scalar * r

    // pt.x * r
    // pt.y * r
    // pt.t = montgomery_product(xr, yr)
    // pt.z = r
}
