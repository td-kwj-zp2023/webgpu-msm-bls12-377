{{> curve_functions }}

@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;

@group(0) @binding(2)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(3)
var<storage, read> points: array<Point>;

const NUM_ROWS = {{ num_rows }}u;
const MAX_COL_IDX = {{ max_col_idx }}u;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    for (var i = 0u; i < arrayLength(&output); i ++) {
        var y: BigInt = get_r();
        var z: BigInt = get_r();

        var inf: Point;
        inf.y = y;
        inf.z = z;
        output[global_id.x + i] = inf;
    }

    let row_begin = row_ptr[global_id.x + i];
    let row_end = row_ptr[global_id.x + i + 1];
    let sum = output[global_id.x + i];

    for (let j = row_begin; j < row_end; j++) {
        var res = add_points(col_idx, sum);

        output[global_id.x + i] = res;
    }
}