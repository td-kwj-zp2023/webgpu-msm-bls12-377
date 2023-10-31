import { create_ell_sparse_matrices_from_points, gen_add_to, merge_points } from './create_ell_from_points'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../reference/utils/FieldMath";
import { to_words_le, from_words_le, genRandomFieldElement, extPointTypeToBigIntPoint } from './utils'
import { CSRSparseMatrix } from './matrices/matrices'

const word_size = 13
const num_words = 20
const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
const fieldMath = new FieldMath();

// Generate input points. These are not random for dev purposes.
const create_test_points = (num_inputs: number) => {
    const points: ExtPointType[] = []

    const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246')
    const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166')
    const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023')
    const z = BigInt('1')
    const pt = fieldMath.createPoint(x, y, t, z)

    for (let i = 0; i < num_inputs; i ++) {
        points.push(pt)
        //points.push(pt.multiply(BigInt(i + 1)))
    }

    return points//.map((p) => { return { x: p.ex, y: p.ey, t: p.et, z: p.ez } })
}

// The performance of this algorithm is not great
const naive_gen_add_to = (chunks: number[]): { add_to: number[], new_chunks: number[] } => {
    const add_to: number[] = []
    const new_chunks: number[] = []

    for (let i = 0; i < chunks.length; i ++) {
        let val = 0
        if (chunks[i] > 0) {
            for (let j = i + 1; j < chunks.length; j ++) {
                if (chunks[i] === chunks[j]) {
                    val = j
                    break
                }
            }
        }
        add_to.push(val)

        if (chunks.slice(i + 1).includes(chunks[i])) {
            new_chunks.push(0)
        } else {
            new_chunks.push(chunks[i])
        }
    }
    return { add_to, new_chunks }
}

const gen_random_test_case = (num_scalar_chunks: number, max: number) => {
    const scalar_chunks: number[] = []
    for (let i = 0; i < num_scalar_chunks; i ++) {
        const r = Math.floor(Math.random() * max)
        scalar_chunks.push(r)
    }
    return scalar_chunks
}

describe('Create an ELL sparse matrix from the MSM input points and scalars', () => {
    describe('generate ELL sparse matrices from MSM inputs', () => {
        it('static example test case', async () => {
            // TODO: we need to refactor matrices.ts
            const num_inputs = 8
            const ext_points = create_test_points(num_inputs)
            const points = ext_points.map(extPointTypeToBigIntPoint)
            const scalars = [4, 0, 7, 3, 0, 3, 4, 3].map((x) => BigInt(x))
            const num_threads = 4
            const ell_sms = create_ell_sparse_matrices_from_points(points, scalars, num_threads)
            const ell_sm = ell_sms[0]
            /*
             * (P_0 + P_6) * 4 +
             * (P_2) * 7 +
             * (P_3 + P_5 + P_7) * 3
             */
            ell_sm.data = ell_sm.data.map((x) => x.map(extPointTypeToBigIntPoint))
            const csr_sm = await (new CSRSparseMatrix([], [], [])).ell_to_csr_sparse_matrix(ell_sm)
            const vp = await csr_sm.smtvp([1, 1, 1, 1].map((x) => BigInt(x)))

            const pt_3 = ext_points[3].add(ext_points[5]).add(ext_points[7])
            const pt_4 = ext_points[0].add(ext_points[6])
            const pt_7 = ext_points[0].add(ext_points[6])

            expect(vp[3] === pt_3)
            expect(vp[4] === pt_4)
            expect(vp[7] === pt_7)
        })

        it('random test cases', () => {
            const num_inputs = 65536
            const points = create_test_points(num_inputs)
            const scalars: bigint[] = []
            for (let i = 0; i < num_inputs; i ++) {
                scalars.push(genRandomFieldElement(p))
            }
            const bigintAffinePoints = []
            for (const point of points) {
                const p = point.toAffine()

                bigintAffinePoints.push({
                    x: p.x,
                    y: p.y,
                    t: fieldMath.Fp.mul(p.x, p.y),
                    z: BigInt(1),
                })
            }

            const num_threads = 16
            const ell_sms = create_ell_sparse_matrices_from_points(bigintAffinePoints, scalars, num_threads)
            //TODO: add checks here
        })
    })

    describe('generate the add_to array', () => {
        it('static example test case', () => {
            const points = create_test_points(20)
            const scalar_chunks = [
                3, 2, 3, 1, 4,
                4, 5, 4, 7, 9,
                1, 0, 1, 0, 1,
                2, 5, 9, 9, 3,
            ]

            const { add_to, new_chunks } = gen_add_to(scalar_chunks)

            const expected_add_to = [
                2, 15, 19, 10, 5,
                7, 16, 0, 0, 17,
                12, 0, 14, 0, 0,
                0, 0, 18, 0, 0
            ]

            const expected_new_chunks = [
                0, 0, 0, 0, 0,
                0, 0, 4, 7, 0,
                0, 0, 0, 0, 1,
                2, 5, 0, 9, 3
            ]

            const e = naive_gen_add_to(scalar_chunks)

            expect(e.add_to).toEqual(add_to)
            expect(e.add_to).toEqual(expected_add_to)
            expect(e.new_chunks).toEqual(new_chunks)
            expect(e.new_chunks).toEqual(expected_new_chunks)

            const ZERO_POINT = fieldMath.createPoint(
                BigInt(0),
                BigInt(1),
                BigInt(0),
                BigInt(1),
            )
            const merged_points = merge_points(points, add_to, ZERO_POINT)
            for (let i = 0; i < merged_points.length; i ++) {
                if (new_chunks[i] === 0) {
                    // Ignore this case because we'll ignore points whose
                    // corresponding scalar chunk equals 0 anyway
                } else {
                    expect(merged_points[i]).not.toEqual(ZERO_POINT)
                }
            }
        })

        it('random test cases', () => {
            const num_tests = 1000
            for (let i = 0; i < num_tests; i ++) {
                const scalar_chunks = gen_random_test_case(num_words, 10)
                const e = naive_gen_add_to(scalar_chunks)
                const { add_to, new_chunks } = gen_add_to(scalar_chunks)
                expect(add_to).toEqual(e.add_to)
                expect(new_chunks).toEqual(e.new_chunks)
            }
        })
    })
})
