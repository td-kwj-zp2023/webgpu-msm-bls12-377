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
    for (let i = 0; i < NZ; i++) {
        row_ptr[col_idx[global_id.x] + 2] += 1;
    }

    for (let i = 2; i < row_ptr.length; i++) {
        // Calculate incremental sum
        row_ptr[i] += row_ptr[i - 1];
    }

    for (let i = 0; i < n; i++) {
        for (let j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            const new_index = row_ptr[col_idx[j] + 1];
            row_ptr[col_idx[j] + 1] += 1;
            sparse_matrix[new_index] = points[j];
            col_idx[new_index] = i;
        }
    }
    // output[global_id.x] = add_points(points[global_id.x], points[global_id.x]);
}