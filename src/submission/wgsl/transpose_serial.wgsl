// Input buffers
@group(0) @binding(0)
var<storage, read> all_csr_col_idx: array<u32>;

// Output buffers
@group(0) @binding(1)
var<storage, read_write> all_csc_col_ptr: array<u32>;
@group(0) @binding(2)
var<storage, read_write> all_csc_val_idxs: array<u32>;

// Intermediate buffer
@group(0) @binding(3)
var<storage, read_write> all_curr: array<u32>;

// Params buffer
@group(0) @binding(4)
var<uniform> params: vec3<u32>;

fn calc_start_end(m: u32, n: u32, i: u32) -> vec2<u32> {
    if (i < m) {
        return vec2(i * n, i * n + n);
    } else {
        return vec2(m * n, m * n);
    }
}

@compute
@workgroup_size({{ num_workgroups }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    // Serial transpose algo from
    // https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf

    let subtask_idx = global_id.x;

    // Number of columns
    let m = params[0];

    // Number of rows
    let n = params[1];

    // Input size
    let input_size = params[2];

    let ccp_offset = subtask_idx * (n + 1u);
    let cci_offset = subtask_idx * input_size;
    let curr_offset = subtask_idx * n;

    for (var i = 0u; i < m; i ++) {
        let r = calc_start_end(m, n, i);
        let start = r[0];
        let end = r[1];
        for (var j = start; j < end; j ++) {
            all_csc_col_ptr[
                ccp_offset + all_csr_col_idx[cci_offset + j] + 1u
            ] += 1u;
            storageBarrier();
        }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (var i = 1u; i < n + 1u; i ++) {
        all_csc_col_ptr[ccp_offset + i] +=
            all_csc_col_ptr[ccp_offset + i - 1u];
        storageBarrier();
    }

    var val = 0u;
    for (var i = 0u; i < m; i ++) {
        let r = calc_start_end(m, n, i);
        let start = r[0];
        let end = r[1];
        for (var j = start; j < end; j ++) {
            let loc = all_csc_col_ptr[
                ccp_offset + all_csr_col_idx[cci_offset + j]
            ] + all_curr[
                curr_offset + all_csr_col_idx[cci_offset + j]
            ];

            all_curr[curr_offset + all_csr_col_idx[cci_offset + j]] ++;

            all_csc_val_idxs[cci_offset + loc] = val;
            val ++;
        }
    }
}

