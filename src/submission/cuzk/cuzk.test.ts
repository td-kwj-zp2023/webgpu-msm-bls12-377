jest.setTimeout(10000000)
import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { decompose_scalars } from '../../submission/utils'
import { cpu_transpose } from '../../submission/cuzk/transpose'
import { cpu_smvp } from '../../submission/cuzk/smvp';

const fieldMath = new FieldMath()
const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246')
const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166')
const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023')
const z = BigInt('1')
const pt = fieldMath.createPoint(x, y, t, z)

describe('cuzk', () => {
    it('one subtask', () => {
        const input_size = 32
        const chunk_size = 4

        const num_columns = 2 ** chunk_size
        const num_rows_per_subtask = input_size

        const num_chunks_per_scalar = Math.ceil(256 / chunk_size)
        const num_subtasks = num_chunks_per_scalar

        const points: ExtPointType[] = []
        const scalars: bigint[] = []
        for (let i = 0; i < input_size; i ++) {
            const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
            const v = BigInt('0x1a1b13111111112c11b8111113331111711111f1111111111111311131131121111811171111')
            scalars.push((BigInt(i) * v) % p)
            points.push(fieldMath.createPoint(x, y, t, z))
        }

        const decomposed_scalars = decompose_scalars(scalars, num_chunks_per_scalar, chunk_size)

        const bucket_sums: ExtPointType[] = []
        for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
            const subtask_chunks = decomposed_scalars[subtask_idx]

            // Copy subtask_chunks to col_idx
            const col_idx = subtask_chunks.map((x) => Number(x))

            // Construct row_ptr
            let j = 0
            const row_ptr: number[] = [0]
            for (let i = 0; i < subtask_chunks.length; i += num_columns) {
                row_ptr.push(row_ptr[j] + num_columns)
                j ++
            }

            const {
                csc_col_ptr,
                csc_row_idx,
                csc_vals,
            }  = cpu_transpose(row_ptr, col_idx, num_columns)

            // SMVP
            const buckets = cpu_smvp(csc_col_ptr, points, fieldMath)

            let bucket_sum = fieldMath.customEdwards.ExtendedPoint.ZERO
            for (let i = 1; i < buckets.length; i ++) {
                const b = buckets[i].multiply(BigInt(i))
                bucket_sum = bucket_sum.add(b)
            }
            bucket_sums.push(bucket_sum)
        }

        // Horner's rule
        const m = BigInt(2) ** BigInt(chunk_size)
        // The last scalar chunk is the most significant digit (base m)
        let result = bucket_sums[bucket_sums.length - 1]
        for (let i = bucket_sums.length - 2; i >= 0; i --) {
            result = result.multiply(m)
            result = result.add(bucket_sums[i])
        }

        const result_affine = result.toAffine()
        console.log(result_affine)

        // Calculated expected result
        let expected = fieldMath.customEdwards.ExtendedPoint.ZERO
        for (let i = 0; i < input_size; i ++) {
            if (scalars[i] !== BigInt(0)) {
                const p = points[i].multiply(scalars[i])
                expected = expected.add(p)
            }
        }
        const expected_affine = expected.toAffine()
        console.log(expected_affine)
    })
})

export {}
