{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> ec_funcs }}
{{> curve_parameters }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> points: array<Point>;
@group(0) @binding(1)
var<storage, read> scalars: array<u32>;
@group(0) @binding(2)
var<storage, read_write> results: array<Point>;

@compute
@workgroup_size({{ workgroup_size }})
fn blah2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
}

// Double-and-add algo from the ZPrize test harness
fn double_and_add(point: Point, scalar: u32) -> Point {
    // Set result to the point at infinity
    var result: Point;
    result.y = get_r();
    result.z = get_r();

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

@compute
@workgroup_size({{ workgroup_size }})
fn double_and_add_benchmark(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    var point = points[id];
    var scalar = scalars[id];

    var result = point;

    let cost = {{ cost }}u;

    for (var i = 0u; i < cost; i ++) {
        result = double_and_add(result, scalar);
    }
    results[id] = result;
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

// Booth encoding method
fn booth(point: Point, scalar: u32) -> Point {
    if (scalar == 0u) {
        return point;
    }

    // Binary decomposition of the scalar
    var a: array<u32, 17>;

    var s = scalar;
    var i = 0u;
    while (s != 0u) {
        a[i] = s & 1u;
        s = s >> 1u;
        i ++;
    }

    for (var i = 16u; i >= 1u; i --) {
        if (a[i] == 0u && a[i - 1u] == 1u) {
            a[i] = 1u;
        } else if (a[i] == 1u && a[i - 1u] == 0u) {
            a[i] = 2u;
        //} else if (a[i] == 0u && a[i - 1u] == 0u) {
            ////a[i] = 0
        } else if (a[i] == 1u && a[i - 1u] == 1u) {
            a[i] = 0u;
        }
    }

    if (a[0] == 1u) {
        a[0] = 2u;
    }

    // Find the last 1
    var max_idx = 16u;
    while (a[max_idx] == 0u) {
        max_idx --;
    }

    // Set result to the point at infinity
    var result: Point;
    result.y = get_r();
    result.z = get_r();

    var temp = point;
    for (var i = 0u; i < max_idx + 1u; i ++) {
        if (a[i] == 1u) {
            result = add_points(result, temp);
        } else if (a[i] == 2u) {
            result = add_points(result, negate_point(temp));
        }
        temp = double_point(temp);
    }

    return result;
}

@compute
@workgroup_size({{ workgroup_size }})
fn booth_benchmark(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    var point = points[id];
    var scalar = scalars[id];

    var result = point;

    let cost = {{ cost }}u;

    for (var i = 0u; i < cost; i ++) {
        result = booth(result, scalar);
    }
    results[id] = result;
}
