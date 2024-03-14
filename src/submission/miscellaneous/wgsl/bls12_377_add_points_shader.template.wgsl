{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> x_coords: array<BigInt>;
@group(0) @binding(1)
var<storage, read> y_coords: array<BigInt>;
@group(0) @binding(2)
var<storage, read_write> out_x_coords: array<BigInt>;
@group(0) @binding(3)
var<storage, read_write> out_y_coords: array<BigInt>;
@group(0) @binding(4)
var<storage, read_write> out_z_coords: array<BigInt>;

struct Point {
  x: BigInt,
  y: BigInt,
  z: BigInt
}

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var id = global_id.x * 2u;

    var x = x_coords[id];
    var y = y_coords[id];
    var z: BigInt = get_r();
    var pt = Point(x, y, z);

    x = x_coords[id + 1u];
    y = y_coords[id + 1u];
    var pt2 = Point(x, y, z);

    // pt + pt2
    var added = add_points(pt, pt2);
    out_x_coords[id] = added.x;
    out_y_coords[id] = added.y;
    out_z_coords[id] = added.z;

    // dbl(pt + pt2)
    var doubled = double_point(add_points(pt, pt2));
    out_x_coords[id + 1u] = doubled.x;
    out_y_coords[id + 1u] = doubled.y;
    out_z_coords[id + 1u] = doubled.z;
}
