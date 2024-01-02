jest.setTimeout(10000000)
import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { decompose_scalars_signed } from '../../submission/utils'
import { cpu_transpose } from '../../submission/cuzk/transpose'
import { cpu_smvp_signed } from '../../submission/cuzk/smvp';

const fieldMath = new FieldMath()
const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246')
const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166')
const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023')
const z = BigInt('1')
const pt = fieldMath.createPoint(x, y, t, z)

describe('cuzk', () => {
    // TODO: cuzk with precomputation
 
    it('cuzk without precomputation', () => {
        const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

        const input_size = 4
        const chunk_size = 4

        const num_columns = 2 ** chunk_size

        const num_chunks_per_scalar = Math.ceil(256 / chunk_size)
        const num_subtasks = num_chunks_per_scalar

        const points: ExtPointType[] = []
        const scalars: bigint[] = []
        for (let i = 0; i < input_size; i ++) {
            points.push(pt)
            const v = BigInt('1111111111111111111111111111111111111111111111111111111111111111111111111111')
            scalars.push((BigInt(i) * v) % p)
            points.push(fieldMath.createPoint(x, y, t, z).multiply(BigInt(i + 1)))
        }

        const decomposed_scalars = decompose_scalars_signed(scalars, num_subtasks, chunk_size)

        const bucket_sums: ExtPointType[] = []
        for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
            const subtask_chunks = decomposed_scalars[subtask_idx]

            // Copy subtask_chunks to col_idx
            const col_idx = subtask_chunks.map((x) => Number(x))

            // Construct row_ptr
            let j = 0
            const row_ptr: number[] = [0]
            for (let i = 0; i < input_size; i += num_columns) {
                row_ptr.push(row_ptr[j] + num_columns)
                j ++
            }
            while (row_ptr.length < input_size + 1) {
                row_ptr.push(row_ptr[row_ptr.length - 1])
            }

            const { csc_col_ptr, csc_vals } =
                cpu_transpose(row_ptr, col_idx, num_columns)

            // Perform SMVP
            const buckets = cpu_smvp_signed(
                csc_col_ptr,
                csc_vals,
                points,
                chunk_size,
                fieldMath,
            )

            let bucket_sum = fieldMath.customEdwards.ExtendedPoint.ZERO
            for (let i = 0; i < buckets.length; i ++) {
                if (!buckets[i].equals(fieldMath.customEdwards.ExtendedPoint.ZERO)) {
                    bucket_sum = bucket_sum.add(buckets[i])
                }
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

        // Calculated expected result
        let expected = fieldMath.customEdwards.ExtendedPoint.ZERO
        for (let i = 0; i < input_size; i ++) {
            if (scalars[i] !== BigInt(0)) {
                const p = points[i].multiply(scalars[i])
                expected = expected.add(p)
            }
        }
        const expected_affine = expected.toAffine()
        expect(result_affine.x).toEqual(expected_affine.x)
        expect(result_affine.y).toEqual(expected_affine.y)
    })
})

export {}
