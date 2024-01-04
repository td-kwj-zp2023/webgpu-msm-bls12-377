// Input buffers
@group(0) @binding(0)
var<storage, read> csr_col_idx: array<u32>;

// Output buffers
@group(0) @binding(1)
var<storage, read_write> csc_col_ptr: array<u32>;
@group(0) @binding(2)
var<storage, read_write> csc_val_idxs: array<u32>;

// Intermediate buffer
@group(0) @binding(3)
var<storage, read_write> curr: array<u32>;
@group(0) @binding(4)
var<uniform> params: vec2<u32>;

fn calc_start_end(m: u32, n: u32, i: u32) -> vec2<u32> {
    if (i < m) {
        return vec2(i * n, i * n + n);
    } else {
        return vec2(m * n, m * n);
    }
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    // Serial transpose algo from
    // https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf

    // Number of columns
    let m = params[0];

    // Number of rows
    let n = params[1];

    for (var i = 0u; i < m; i ++) {
        let r = calc_start_end(m, n, i);
        let start = r[0];
        let end = r[1];
        for (var j = start; j < end; j ++) {
            csc_col_ptr[csr_col_idx[j] + 1u] += 1u;
            storageBarrier();
        }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (var i = 1u; i < n + 1u; i ++) {
        csc_col_ptr[i] += csc_col_ptr[i - 1u];
        storageBarrier();
    }

    var val = 0u;
    for (var i = 0u; i < m; i ++) {
        let r = calc_start_end(m, n, i);
        let start = r[0];
        let end = r[1];
        for (var j = start; j < end; j ++) {
            let loc = csc_col_ptr[csr_col_idx[j]] + curr[csr_col_idx[j]];
            curr[csr_col_idx[j]] ++;
            csc_val_idxs[loc] = val;
            val ++;
        }
    }
}

