@group(0) @binding(0)
var<storage, read> csr_row_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> csr_col_idx: array<u32>;
@group(0) @binding(2)
var<storage, read_write> csc_col_ptr: array<u32>;
@group(0) @binding(3)
var<storage, read_write> csc_row_idx: array<u32>;
@group(0) @binding(4)
var<storage, read_write> csc_val_idxs: array<u32>;
@group(0) @binding(5)
var<storage, read_write> curr: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    // Number of rows
    let m = arrayLength(&csr_row_ptr) - 1u;

    // Number of columns
    let n = {{ num_cols }}u;

    for (var i = 0u; i < m; i ++) {
        for (var j = csr_row_ptr[i]; j < csr_row_ptr[i + 1u]; j ++) {
            csc_col_ptr[csr_col_idx[j] + 1u] += 1u;
        }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (var i = 1u; i < n + 1u; i ++) {
        csc_col_ptr[i] += csc_col_ptr[i - 1u];
    }

    var val = 0u;
    for (var i = 0u; i < m; i ++) {
        for (var j = csr_row_ptr[i]; j < csr_row_ptr[i + 1u]; j ++) {
            let loc = csc_col_ptr[csr_col_idx[j]] + curr[csr_col_idx[j]];
            csc_row_idx[loc] = i;
            curr[csr_col_idx[j]] ++;
            csc_val_idxs[loc] = val;
            val ++;
        }
    }
}

