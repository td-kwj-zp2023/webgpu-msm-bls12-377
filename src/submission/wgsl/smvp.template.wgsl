{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}
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

// Params buffer
@group(0) @binding(8)
var<uniform> params: vec3<u32>;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

fn get_paf() -> Point {
    var result: Point;
    let r = get_r();
    result.y = r;
    result.z = r;
    return result;
}

// This double-and-add code is adapted from the ZPrize test harness:
// https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/webgpu/wgsl/Curve.ts#L78
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

// Point negation only involves multiplying the X and T coordinates by -1 in
// the field
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
    let input_size = params[0];
    let num_y_workgroups = params[1];
    let num_z_workgroups = params[2];

    let gidx = global_id.x; 
    let gidy = global_id.y; 
    var gidz = global_id.z;
    let id = (gidx * num_y_workgroups + gidy) * num_z_workgroups + gidz;

    let num_columns = {{ num_columns }}u;
    let l = num_columns;
    let h = {{ half_num_columns }}u;

    // Define custom subtask_idx
    let subtask_idx = (id / h);

    var inf = get_paf();

    let rp_offset = subtask_idx * (num_columns + 1u);

    // As we use the signed bucket index technique, each thread handles two
    // buckets.
    for (var j = 0u; j < 2u; j ++) {
        var row_idx = (id % h) + h;
        if (j == 1u) {
            row_idx = (id % h);
        }

        let row_begin = row_ptr[rp_offset + row_idx];
        let row_end = row_ptr[rp_offset + row_idx + 1u];
        var sum = inf;

        // Add up all the points in the bucket
        for (var k = row_begin; k < row_end; k ++) {
            let idx = val_idx[subtask_idx * input_size + k];

            var x = new_point_x[idx];
            var y = new_point_y[idx];

            // We didn't compute the t and z coordiantes in the previous shader
            // because there is a limit to the number of buffers that may be
            // bound to a shader, so we do so here. Fortunately the computation
            // is relatively simple: t = xyr and z = 1r.
            var t = montgomery_product(&x, &y);
            var z = get_r();

            let pt = Point(x, y, t, z);
            sum = add_points(sum, pt);
        }

        // Negate the point if the recovered bucket index is negative.
        // Since we've added half_num_columns to each scalar chunk in
        // convert_point_coords_and_decompose_scalars.template.wgsl, we know if
        // the original bucket index is negative if it is less than
        // half_num_columns.
        var bucket_idx = 0u;
        if (h > row_idx) {
            bucket_idx = h - row_idx;
            sum = negate_point(sum);
        } else {
            bucket_idx = row_idx - h;
        }

        // Multiply the bucket sum by the scalar chunk / bucket ID
        sum = double_and_add(sum, bucket_idx);

        // Store the result in buckets[thread_id]. Each thread must use
        // a unique storage location (thread_id) to prevent race
        // conditions.
        if (j == 1) {
            // Since the point has been set, add to it.
            let bucket = Point(
                bucket_sum_x[id],
                bucket_sum_y[id],
                bucket_sum_t[id],
                bucket_sum_z[id]
            );
            sum = add_points(bucket, sum);
        }
        // Set the point. Since no point has been set when j == 0, we can just
        // overwrite the data.
        bucket_sum_x[id] = sum.x;
        bucket_sum_y[id] = sum.y;
        bucket_sum_z[id] = sum.z;
        bucket_sum_t[id] = sum.t;
    }

    {{{ recompile }}}
}
