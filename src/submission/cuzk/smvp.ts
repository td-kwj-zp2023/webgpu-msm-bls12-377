import { G1 } from '@celo/bls12377js'
import {
    //createAffinePoint,
    scalarMult,
    ZERO,
} from '../bls12_377'

// Perform SMVP and scalar mul with signed bucket indices.
export const cpu_smvp_signed = (
    subtask_idx: number,
    input_size: number,
    num_columns: number,
    chunk_size: number,
    all_csc_col_ptr: number[],
    all_csc_val_idxs: number[],
    points: G1[],
) => {
    const l = 2 ** chunk_size
    const h = l / 2
    const zero = ZERO

    const buckets: G1[] = [];
    for (let i = 0; i < num_columns / 2 + 1; i++) {
        buckets.push(zero)
    }

    const rp_offset = subtask_idx * (num_columns + 1)

    // In a GPU implementation, each iteration of this loop should be performed by a thread.
    // Each thread handles two buckets
    for (let thread_id = 0; thread_id < num_columns / 2; thread_id ++) {
        const bucket_idxs: number[] = []
        for (let j = 0; j < 2; j ++) {
            // row_idx is the index of the row in the CSR matrix. It is *not*
            // the same as the bucket index.
            let row_idx = thread_id + num_columns / 2
            if (j === 1) {
                row_idx = thread_id
            }

            const row_begin = all_csc_col_ptr[rp_offset + row_idx];
            const row_end = all_csc_col_ptr[rp_offset + row_idx + 1];

            let sum = zero
            for (let k = row_begin; k < row_end; k ++) {
                sum = sum.add(
                    points[all_csc_val_idxs[subtask_idx * input_size + k]]
                )
            }

            let bucket_idx
            if (h > row_idx) {
                bucket_idx = h - row_idx
                try {
                    sum = sum.negate()
                } catch {
                    debugger
                }
            } else {
                bucket_idx = row_idx - h
            }

            //console.log({ thread_id, row_idx, bucket_idx })

            if (bucket_idx > 0) {
                sum = scalarMult(sum, BigInt(bucket_idx))

                // Store the result in buckets[thread_id]. Each thread must use
                // a unique storage location (thread_id) to prevent race
                // conditions.
                buckets[thread_id] = buckets[thread_id].add(sum)
            }

            bucket_idxs.push(bucket_idx)
        }
    }

    return buckets
}
