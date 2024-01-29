jest.setTimeout(10000000)
import { G1 } from '@celo/bls12377js'
import { decompose_scalars_signed } from './utils'
import { cpu_transpose } from './transpose'
import { cpu_smvp_signed } from './smvp';
import { running_sum_bucket_reduction, parallel_bucket_reduction } from './bpr'
import { createAffinePoint, scalarMult, ZERO } from '../bls12_377'

const x = BigInt('111871295567327857271108656266735188604298176728428155068227918632083036401841336689521497731900230387779623820740');
const y = BigInt('76860045326390600098227152997486448974650822224305058012700629806287380625419427989664237630603922765089083164740');
const z = BigInt('1');
const pt = createAffinePoint(x, y, z)

describe('cuzk', () => {
    it('cuzk with smaller inputs and parameters', () => {
        const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

        const input_size = 16
        const chunk_size = 4

        expect(input_size >= 2 ** chunk_size).toBeTruthy()

        const num_columns = 2 ** chunk_size
        const num_rows = Math.ceil(input_size / num_columns)

        const num_chunks_per_scalar = Math.ceil(256 / chunk_size)
        const num_subtasks = num_chunks_per_scalar

        const points: G1[] = []
        const scalars: bigint[] = []
        for (let i = 0; i < input_size; i ++) {
            points.push(pt)
            const v = BigInt('1111111111111111111111111111111111111111111111111111111111111111111111111111')
            scalars.push((BigInt(i) * v) % p)
            points.push(scalarMult(pt, BigInt(i + 1)))
        }

        const decomposed_scalars = decompose_scalars_signed(scalars, num_subtasks, chunk_size)

        const bucket_sums: G1[] = []

        // Perform multiple transpositions "in parallel"
        const { all_csc_col_ptr, all_csc_vals } = cpu_transpose(
            decomposed_scalars.flat(),
            num_columns,
            num_rows,
            num_subtasks,
            input_size,
        )

        const zero = ZERO
        for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
            // Perform SMVP
            const buckets = cpu_smvp_signed(
                subtask_idx,
                input_size,
                num_columns,
                chunk_size,
                all_csc_col_ptr,
                all_csc_vals,
                points,
            )

            const bucket_sum_serial = serial_bucket_reduction(buckets)
            const bucket_sum_rs = running_sum_bucket_reduction(buckets)

            let bucket_sum = zero
            for (const b of parallel_bucket_reduction(buckets)) {
                bucket_sum = bucket_sum.add(b)
            }

            expect(bucket_sum_rs.equals(bucket_sum)).toBeTruthy()
            expect(bucket_sum_serial.equals(bucket_sum)).toBeTruthy()

            bucket_sums.push(bucket_sum)
        }

        // Horner's rule
        const m = BigInt(2) ** BigInt(chunk_size)
        // The last scalar chunk is the most significant digit (base m)
        let result = bucket_sums[bucket_sums.length - 1]
        for (let i = bucket_sums.length - 2; i >= 0; i --) {
            result = scalarMult(result, m)
            result = result.add(bucket_sums[i])
        }

        // Calculated expected result
        let expected = zero
        for (let i = 0; i < input_size; i ++) {
            if (scalars[i] !== BigInt(0)) {
                expected = expected.add(scalarMult(points[i], scalars[i]))
            }
        }
        expect(result.equals(expected)).toBeTruthy()
    })
})

const serial_bucket_reduction = (buckets: G1[]) => {
    const indices = []
    for (let i = 1; i < buckets.length; i ++) {
        indices.push(i)
    }
    indices.push(0)

    let bucket_sum = ZERO

    for (let i = 1; i < buckets.length + 1; i++) {
        const b = scalarMult(buckets[indices[i - 1]], BigInt(i))
        bucket_sum = bucket_sum.add(b)
        //console.log(`serial: buckets[${indices[i - 1]}] * ${i}`)
    }
    return bucket_sum
}


export {}
