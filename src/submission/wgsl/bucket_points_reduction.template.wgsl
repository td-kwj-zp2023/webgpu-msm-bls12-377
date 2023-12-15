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
@workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var gidx = global_id.x;
    var gidy = global_id.y;
    let id = gidx * 256u + gidy;

    let a = id * 4u;
    // Given points A and B, point_x_y looks like [Ax, Ay, Bx, By]
    // Given points A and B, point_t_z looks like [At, Az, Bt, Bz]

    let a1 = a + 1u;
    let a_x = point_x_y[a];
    let a_y = point_x_y[a1];
    let a_t = point_t_z[a];
    let a_z = point_t_z[a1];

    let b = a + 2u;
    let b1 = b + 1u;
    let b_x = point_x_y[b];
    let b_y = point_x_y[b1];
    let b_t = point_t_z[b];
    let b_z = point_t_z[b1];
    var pt_b = Point(b_x, b_y, b_t, b_z);

    // In case the number of points is odd, assign the point at infinity to B
    if (num_points % 2u == 1u && b >= num_points * 2u) {
        pt_b = get_paf();
    } 

    let pt_a = Point(a_x, a_y, a_t, a_z);
    let result = add_points(pt_a, pt_b);

    let id2 = id * 2u;
    let id2_1 = id2 + 1;
    out_x_y[id2] = result.x;
    out_x_y[id2_1] = result.y;
    out_t_z[id2] = result.t;
    out_t_z[id2_1] = result.z;
}
