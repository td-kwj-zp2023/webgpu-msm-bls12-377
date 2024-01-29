{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}
{{> ec_funcs }}

// Input buffers
@group(0) @binding(0)
var<storage, read> bucket_sum_x: array<BigInt>;
@group(0) @binding(1)
var<storage, read> bucket_sum_y: array<BigInt>;
@group(0) @binding(2)
var<storage, read> bucket_sum_t: array<BigInt>;
@group(0) @binding(3)
var<storage, read> bucket_sum_z: array<BigInt>;

// Output buffers
@group(0) @binding(4)
var<storage, read_write> reduced_bucket_x: array<BigInt>;
@group(0) @binding(5)
var<storage, read_write> reduced_bucket_y: array<BigInt>;
@group(0) @binding(6)
var<storage, read_write> reduced_bucket_t: array<BigInt>;
@group(0) @binding(7)
var<storage, read_write> reduced_bucket_z: array<BigInt>;

// Params buffer
@group(0) @binding(8)
var<uniform> params: vec2<u32>;

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

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    let thread_id = global_id.x; 
    let num_threads = {{ workgroup_size }}u;

    let subtask_idx = params[0];
    let num_columns = params[1];

    // Number of buckets per subtask
    let n = num_columns / 2u;

    // Number of buckets to reduce per thread
    let buckets_per_thread = n / num_threads;

    let bucket_sum_offset = n * subtask_idx;

    var idx = bucket_sum_offset; 
    if (thread_id != 0u) {
        idx = (num_threads - thread_id) * buckets_per_thread + bucket_sum_offset;
    }

    var m = Point(
        bucket_sum_x[idx],
        bucket_sum_y[idx],
        bucket_sum_t[idx],
        bucket_sum_z[idx]
    );
    var g = m;

    for (var i = 0u; i < buckets_per_thread - 1u; i ++) {
        let idx = (num_threads - thread_id) * buckets_per_thread - 1u - i;
        let bi = bucket_sum_offset + idx;
        let b = Point(
            bucket_sum_x[bi],
            bucket_sum_y[bi],
            bucket_sum_t[bi],
            bucket_sum_z[bi]
        );
        m = add_points(m, b);
        g = add_points(g, m);
    }

    let s = buckets_per_thread * (num_threads - thread_id - 1u);
    if (s > 0u) {
        g = add_points(g, double_and_add(m, s));
    }

    let t = subtask_idx * num_threads + thread_id;
    reduced_bucket_x[t] = g.x;
    reduced_bucket_y[t] = g.y;
    reduced_bucket_t[t] = g.t;
    reduced_bucket_z[t] = g.z;

    {{{ recompile }}}
}
