{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> curve_parameters }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> point_x_y: array<BigInt>;
@group(0) @binding(1)
var<storage, read> point_t_z: array<BigInt>;
@group(0) @binding(2)
var<storage, read> num_points: u32;

@group(0) @binding(3)
var<storage, read_write> out_x_y: array<BigInt>;
@group(0) @binding(4)
var<storage, read_write> out_t_z: array<BigInt>;

fn get_paf() -> Point {
    var result: Point;
    let r = get_r();
    result.y = r;
    result.z = r;
    return result;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var gidx = global_id.x;
    var gidy = global_id.y;
    let id = gidx * 256u + gidy;

    let a = id * 4u;

    let a_x = point_x_y[a + 0];
    let a_y = point_x_y[a + 1];
    let a_t = point_t_z[a + 0];
    let a_z = point_t_z[a + 1];

    /*
    num_points = 7
    0 1 2 3 4 5 6 7 8 9 10 11 12 13
    id = 0; a = 0,  b = 2;
    id = 1; a = 4;  b = 6;
    id = 2; a = 8;  b = 10;
    id = 2; a = 12; b = 14;
    */

    let b = a + 2u;
    let b_x = point_x_y[b + 0];
    let b_y = point_x_y[b + 1];
    let b_t = point_t_z[b + 0];
    let b_z = point_t_z[b + 1];
    var pt_b = Point(b_x, b_y, b_t, b_z);

    if (num_points % 2u == 1u && b >= num_points * 2u) {
        pt_b = get_paf();
    } 

    let pt_a = Point(a_x, a_y, a_t, a_z);
    let result = add_points(pt_a, pt_b);

    out_x_y[(id * 2)    ] = result.x;
    out_x_y[(id * 2) + 1] = result.y;
    out_t_z[(id * 2)    ] = result.t;
    out_t_z[(id * 2) + 1] = result.z;
}
