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
fn double_and_add(point: Point, s: u32) -> Point {
    var result: Point;
    result.y = get_r();
    result.z = get_r();

    var scalar = s;
    var temp = point;

    let cost = {{ cost }}u;

    while (scalar != 0u) {
        if ((scalar & 1u) == 1u) {
            result = add_points(result, temp);
        }

        temp = add_points(temp, temp);

        scalar = scalar >> 1u;
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
