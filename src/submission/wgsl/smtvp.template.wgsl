struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

struct ETEPoint {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}

const NUM_ROWS = {{ num_rows }}u;

@group(0) @binding(0)
var<storage, read_write> output: array<ETEPoint>;

@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;

@group(0) @binding(2)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(3)
var<storage, read> points: array<ETEPoint>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assumes that the output buffer already contains the point at infinity at
    // each index

    /*
    // Perform SMTVP
    for (var i = 0u; i < NUM_ROWS; i ++) {
        let row_start = row_ptr[global_id.x + i];
        let row_end = row_ptr[global_id.x + i + 1];
        for (var j = row_start; j < row_end; j ++) {
            // temp = self.data[jj] * scalar_vector[row_i]
            // col = self.col_indices[jj]
            // y[col] += temp
        }
    }
    */
    /*output[global_id.x] = points[global_id.x];*/
    output[global_id.x] = output[global_id.x];
}
