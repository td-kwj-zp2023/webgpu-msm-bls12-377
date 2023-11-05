{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}
{{> curve_parameters }}
{{> curve_functions }}

@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(2)
var<storage, read> points: array<Point>;

@group(0) @binding(3)
var<storage, read_write> loop_index: u32;

const N = {{ NUM_ROWS_GPU }}u;

@compute
@workgroup_size(N)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    if (loop_index == 0u) {
        for (var i = 0u; i < arrayLength(&output); i++) {
            var y: BigInt = get_r();
            var z: BigInt = get_r();

            var inf: Point;
            inf.y = y;
            inf.z = z;
            output[global_id.x + i] = inf;
        }
    }

    if (global_id.x < arrayLength(&output)) {
        let row_begin = row_ptr[global_id.x];
        let row_end = row_ptr[global_id.x + 1];
        var sum = output[global_id.x];
        for (var j = row_begin; j < row_end; j++) {
            sum = add_points(points[j], sum);
        }
        output[global_id.x] = sum;
    }
}