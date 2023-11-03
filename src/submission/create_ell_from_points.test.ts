import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../reference/utils/FieldMath";
import { genRandomFieldElement, decompose_scalars } from './utils'
import { prep_for_sort_method, prep_for_cluster_method, pre_aggregate_cpu } from './create_ell'

const word_size = 13
const num_words = 20
const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
const fieldMath = new FieldMath();
const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246')
const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166')
const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023')
const z = BigInt('1')
const pt = fieldMath.createPoint(x, y, t, z)

// Generate input points.
const create_test_points = (num_inputs: number, different_points=false) => {
    const points: ExtPointType[] = []
    for (let i = 0; i < num_inputs; i ++) {
        if (different_points) {
            points.push(pt.multiply(BigInt(i + 1)))
        } else {
            points.push(pt)
        }
    }

    return points
}

const create_test_scalars = (num_inputs: number) => {
    const scalars: bigint[] = []
    for (let i = 0; i < num_inputs; i ++) {
        scalars.push(genRandomFieldElement(p))
    }
    return scalars
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

// Sanity-check the output of the prep functions
const check_new_point_indices = (
    scalar_chunks: number[],
    cluster_start_indices: number[],
    new_point_indices: number[],
) => {
    expect(new_point_indices.length === scalar_chunks.length)

    // Check that new_point_indices and scalar_chunks have the same
    // elements even if in different order
    const n = scalar_chunks.map((x) => x)
    const m = new_point_indices.map((x) => x)
    n.sort()
    m.sort()
    expect(n.toString() === m.toString())

    // Check that the values in cluster_start_indices point to incrementing values in new_point_indices
    for (let i = 0; i < cluster_start_indices.length; i ++) { 
        const start = cluster_start_indices[i]
        const end = i < cluster_start_indices.length - 1 ?
            cluster_start_indices[i + 1] 
            :
            cluster_start_indices[cluster_start_indices.length - 1]
        
        expect(start <= end).toEqual(true)

        let prev = scalar_chunks[new_point_indices[start]]
        for (let j = start + 1; j < end; j ++) {
            expect(scalar_chunks[new_point_indices[j]]).toEqual(prev)
            prev = scalar_chunks[new_point_indices[j]]
        }
    }
}

import * as fs from 'fs'
import * as path from 'path'

describe('Create an ELL sparse matrix from the MSM input points and scalars', () => {
    const num_inputs = 65536
    //const points: ExtPointType[] = []
    let points: ExtPointType[]
    let scalars: bigint[]
    const do_check = false

    beforeAll(() => {
        points = create_test_points(num_inputs)
        scalars = create_test_scalars(num_inputs)

        /*
        const pts_file = fs.readFileSync(
            path.join(
                __dirname,
                '../../public/test-data/points/16-power-points.txt',
            ),
        ).toString()
        for (const line of pts_file.split('\n')) {
            const data = JSON.parse(line)
            const pt = fieldMath.createPoint(
                BigInt(data.x),
                BigInt(data.y),
                BigInt(data.t),
                BigInt(data.z),
            )
            points.push(pt)
        }
        */
    })

    describe('pre-aggregation using the sort method', () => {
        it('small test', () => {
            //scalar_chunks:     [3, 2, 3, 1, 4, 4, 5, 4]
            //new_point_indices: [3, 1, 0, 2, 4, 5, 7, 6]
            //expected:          [0, 1, 2, 4, 7]
            const test_scalar_chunks = [3, 2, 3, 1, 4, 4, 5, 4]
            const test_points = create_test_points(test_scalar_chunks.length, true)

            const r = prep_for_sort_method(test_scalar_chunks, 0, 1)
            expect(r.cluster_start_indices.toString() === [0, 1, 2, 4, 7].toString())
            expect(r.new_point_indices.toString() === [3, 1, 0, 2, 4, 5, 7, 6].toString())

            check_new_point_indices(
                test_scalar_chunks,
                r.cluster_start_indices,
                r.new_point_indices,
            )

            const expected_new_points = [
                test_points[3],
                test_points[1],
                fieldMath.createPoint(BigInt(0), BigInt(0), BigInt(0), BigInt(0)),
                test_points[0].add(test_points[2]),
                fieldMath.createPoint(BigInt(0), BigInt(0), BigInt(0), BigInt(0)),
                fieldMath.createPoint(BigInt(0), BigInt(0), BigInt(0), BigInt(0)),
                test_points[4].add(test_points[5]).add(test_points[7]),
                test_points[6],
            ]

            debugger
            const new_points = pre_aggregate_cpu(
                test_points, 
                r.new_point_indices,
                r.cluster_start_indices,
            )
            expect(new_points.length).toEqual(expected_new_points.length)
            for (let i = 0; i < expected_new_points.length; i ++) {
                expect(new_points[i]).toEqual(expected_new_points[i])
            }
        })

        // Serial performance in Node:
        //   65536 inputs, 16 threads: 675ms
        //   65536 inputs, 8 threads: 656ms
        //   65536 inputs, 1 thread: 660ms
        // Serial performance in browser:
        //   65536 inputs, 16 threads: 277ms
        //   65536 inputs, 8 threads: 346ms
        //   65536 inputs, 1 thread: 3362ms
        // Web worker performance in browser:
        //   65536 inputs, 16 threads: 915ms
        //   65536 inputs, 8 threads: 1019ms
        //   65536 inputs, 1 thread: 1197ms
        it('prep_for_sort_method benchmark', () => {
            // Input: 
            //   - point indices (0 to len(points) - 1)
            //   - scalars
            // Output:
            //   for each decomposed_scalars:
            //     - for each thread:
            //       - new_point_indices[]
            //       - cluster_start_indices[]
            //       The output for each thread can be used to generate a list of
            //       aggregated points and their corresponding scalar chunks

            const num_threads = 16
            const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

            for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
                const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
                for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
                    const { new_point_indices, cluster_start_indices }  = prep_for_sort_method(
                        scalar_chunks,
                        thread_idx,
                        num_threads
                    )
                }
            }
        })

        it('prep and then pre-aggregate benchmark', () => {
            const num_threads = 16
            const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

            for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
                const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
                for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
                    const { new_point_indices, cluster_start_indices }  = prep_for_sort_method(
                        scalar_chunks,
                        thread_idx,
                        num_threads
                    )
                    if (do_check) {
                        check_new_point_indices(
                            scalar_chunks,
                            cluster_start_indices,
                            new_point_indices,
                        )
                    }

                    const new_points = pre_aggregate_cpu(
                        points, 
                        new_point_indices,
                        cluster_start_indices,
                    )
                }
            }
        })
    })

    describe('pre-aggregation using the cluster method', () => {
        it('small test', () => {
            //scalar_chunks:     [3, 2, 3, 1, 4, 4, 5, 4]
            //new_point_indices: [7, 5, 4, 2, 0, 1, 3, 6]
            //expected:          [0, 3, 5, 6, 7]
            const test_scalar_chunks = [3, 2, 3, 1, 4, 4, 5, 4]
            const r = prep_for_cluster_method(test_scalar_chunks, 0, 1)
            check_new_point_indices(
                test_scalar_chunks,
                r.cluster_start_indices,
                r.new_point_indices,
            )
            expect(r.new_point_indices.toString()).toEqual([7, 5, 4, 2, 0, 1, 3, 6].toString())
            expect(r.cluster_start_indices.toString()).toEqual([0, 3, 5, 6, 7].toString())
        })

        // Serial performance in Node:
        //   65536 inputs, 1024 threads: 408ms
        //   65536 inputs, 256 threads: 400ms
        //   65536 inputs, 16 threads: 509ms
        //   65536 inputs, 8 threads: 719ms
        //   65536 inputs, 1 thread: 6567ms
        // Serial performance in browser:
        //   65536 inputs, 16 threads: ms
        //   65536 inputs, 8 threads: ms
        //   65536 inputs, 1 thread: ms
        // Web worker performance in browser:
        //   65536 inputs, 16 threads: 259ms
        //   65536 inputs, 8 threads: 342ms
        //   65536 inputs, 1 thread: 2907
        it('prep_for_cluster_method benchmark', () => {
            // Input: 
            //   - point indices (0 to len(points) - 1)
            //   - scalars
            // Output:
            //   for each decomposed_scalars:
            //     - for each thread:
            //       - new_point_indices[]
            //       - cluster_start_indices[]
            //       The output for each thread can be used to generate a list of
            //       aggregated points and their corresponding scalar chunks
            const num_threads = 16
            const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

            for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
                const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
                for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
                    const { new_point_indices, cluster_start_indices }  = prep_for_cluster_method(
                        scalar_chunks,
                        thread_idx,
                        num_threads,
                    )
                    if (do_check) {
                        check_new_point_indices(
                            scalar_chunks,
                            cluster_start_indices,
                            new_point_indices,
                        )
                    }
                }
            }
        })

        it('prep and then pre-aggregate benchmark', () => {
            const num_threads = 16
            const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

            for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
                const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
                for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
                    const { new_point_indices, cluster_start_indices }  = prep_for_cluster_method(
                        scalar_chunks,
                        thread_idx,
                        num_threads
                    )
                    if (do_check) {
                        check_new_point_indices(
                            scalar_chunks,
                            cluster_start_indices,
                            new_point_indices,
                        )
                    }

                    const new_points = pre_aggregate_cpu(
                        points, 
                        new_point_indices,
                        cluster_start_indices,
                    )
                }
            }
        })
    })

    // create_ell_sparse_matrices_from_points only works in the browser because
    // it uses navigator.hardwareConcurrency and web workers
    /*
    describe('generate ELL sparse matrices from MSM inputs', () => {
        it('static example test case', async () => {
            // TODO: we need to refactor matrices.ts
            const num_inputs = 8
            const ext_points = create_test_points(num_inputs)
            const points = ext_points.map(extPointTypeToBigIntPoint)
            const scalars = [4, 0, 7, 3, 0, 3, 4, 3].map((x) => BigInt(x))
            const num_threads = 4
            const ell_sms = await create_ell_sparse_matrices_from_points(points, scalars, num_threads)
            const ell_sm = ell_sms[0]
            // (P_0 + P_6) * 4 +
            // (P_2) * 7 +
            // (P_3 + P_5 + P_7) * 3
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
    */
})
