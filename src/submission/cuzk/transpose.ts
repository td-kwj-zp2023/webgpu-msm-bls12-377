// Serial transpose algo from
// https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf
export const cpu_transpose = (
    csr_row_ptr: number[],
    csr_col_idx: number[],
    n: number, // The width of the matrix
) => {
    // The height of the matrix
    const m = csr_row_ptr.length - 1

    const curr: number[] = Array(n).fill(0)
    const csc_col_ptr: number[] = Array(n + 1).fill(0)
    const csc_row_idx: number[] = Array(csr_col_idx.length).fill(0)
    const csc_vals: number[] = Array(csr_col_idx.length).fill(0)

    // Calculate the count per column. This step is *not* parallelisable because
    // the index `csr_col_idx[j] + 1` can be the same across iterations,
    // causing a race condition.
    for (let i = 0; i < m; i ++) {
        for (let j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j ++) {
            csc_col_ptr[csr_col_idx[j] + 1] ++
        }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (let i = 1; i < n + 1; i ++) {
        csc_col_ptr[i] += csc_col_ptr[i - 1]
    }

    let val = 0
    for (let i = 0; i < m; i ++) {
        for (let j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j ++) {
            const loc = csc_col_ptr[csr_col_idx[j]] + (curr[csr_col_idx[j]] ++)
            csc_row_idx[loc] = i
            csc_vals[loc] = val
            val ++
        }
    }

    return { csc_col_ptr, csc_row_idx, csc_vals }
}
