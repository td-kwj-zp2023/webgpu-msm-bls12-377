{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}
{{> curve_parameters }}
{{> ec_funcs }}

// Input buffers
@group(0) @binding(0)
var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> val_idx: array<u32>;
@group(0) @binding(2)
var<storage, read> new_point_x: array<BigInt>;
@group(0) @binding(3)
var<storage, read> new_point_y: array<BigInt>;

// Output buffers
@group(0) @binding(4)
var<storage, read_write> bucket_sum_x: array<BigInt>;
@group(0) @binding(5)
var<storage, read_write> bucket_sum_y: array<BigInt>;
@group(0) @binding(6)
var<storage, read_write> bucket_sum_t: array<BigInt>;
@group(0) @binding(7)
var<storage, read_write> bucket_sum_z: array<BigInt>;

fn get_paf() -> Point {
    var result: Point;
    let r = get_r();
    result.y = r;
    result.z = r;
    return result;
}

// Double-and-add algo adapted from the ZPrize test harness
fn double_and_add(point: Point, scalar: u32) -> Point {
    // Set result to the point at infinity
    var result: Point = get_paf();

    var s = scalar;
    var temp = point;

    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = add_points(result, temp);
        }
        temp = double_point(temp);
        s = s >> 1u;
    }
    return result;
}

fn negate_point(point: Point) -> Point {
    var p = get_p();
    var x = point.x;
    var t = point.t;
    var neg_x: BigInt;
    var neg_t: BigInt;
    bigint_sub(&p, &x, &neg_x);
    bigint_sub(&p, &t, &neg_t);
    return Point(neg_x, point.y, neg_t, point.z);
}

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    var inf = get_paf();

    let row_begin = row_ptr[id];
    let row_end = row_ptr[id + 1u];
    var sum = inf;
    for (var j = row_begin; j < row_end; j ++) {
        let idx = val_idx[j];

        var x = new_point_x[idx];
        var y = new_point_y[idx];
        var t = montgomery_product(&x, &y);
        var z = get_r();

        let pt = Point(x, y, t, z);
        sum = add_points(sum, pt);
    }
    let index_shift = {{ index_shift }}u;
    let shifted = id - index_shift;
    if (shifted < index_shift) {
        sum = negate_point(sum);
    }

    sum = double_and_add(sum, shifted);

    bucket_sum_x[id] = sum.x;
    bucket_sum_y[id] = sum.y;
    bucket_sum_t[id] = sum.t;
    bucket_sum_z[id] = sum.z;
}
