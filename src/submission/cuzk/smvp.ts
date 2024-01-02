import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const cpu_smvp = (
    csc_col_ptr: number[],
    csc_val_idxs: number[],
    points: ExtPointType[],
    fieldMath: FieldMath,
) => {
    const currentSum = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const result: ExtPointType[] = [];
    for (let i = 0; i < csc_col_ptr.length - 1; i++) {
        result.push(currentSum)
    }

    for (let i = 0; i < csc_col_ptr.length - 1; i++) {
        const row_begin = csc_col_ptr[i];
        const row_end = csc_col_ptr[i + 1];
        let sum = currentSum;
        for (let j = row_begin; j < row_end; j++) {
            sum = sum.add(points[csc_val_idxs[j]])
        }
        result[i] = sum
    }

    return result
}

// Perform SMVP and scalar mul with signed bucket indices.
export const cpu_smvp_signed = (
    csc_col_ptr: number[],
    csc_val_idxs: number[],
    points: ExtPointType[],
    chunk_size: number,
    fieldMath: FieldMath,
) => {
    const l = 2 ** chunk_size
    const h = l / 2
    const num_columns = csc_col_ptr.length - 1
    const currentSum = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const buckets: ExtPointType[] = [];
    for (let i = 0; i < num_columns / 2; i++) {
        buckets.push(currentSum)
    }

    // In a GPU implementation, each iteration of this loop should be performed by a thread.
    // Each thread handles two buckets:
    // i - h and
    // i + h - l / 2
    for (let idx = 0; idx < num_columns / 2; idx ++) {
        // idx is the thread ID

        for (let j = 0; j < 2; j ++) {
            const i = idx * 2 + j

            const row_begin = csc_col_ptr[i];
            const row_end = csc_col_ptr[i + 1];
            let sum = currentSum
            for (let j = row_begin; j < row_end; j ++) {
                sum = sum.add(points[csc_val_idxs[j]])
            }

            const x = i - h

            let bucket_idx = x
            if (x < 0) {
                bucket_idx = x * -1
                sum = sum.negate()
            }

            const b = bucket_idx - 1
            if (bucket_idx > 0) {
                sum = sum.multiply(BigInt(bucket_idx))
                buckets[b] = buckets[b].add(sum)
            }
        }
    }

    return buckets
}
