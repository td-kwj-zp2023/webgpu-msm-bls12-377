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

            const expected_new_scalar_chunks = [
                test_scalar_chunks[3],
                test_scalar_chunks[1],
                test_scalar_chunks[0],
                test_scalar_chunks[4],
                test_scalar_chunks[6],
            ]

            const expected_new_points = [
                test_points[3],
                test_points[1],
                test_points[0].add(test_points[2]),
                test_points[4].add(test_points[5]).add(test_points[7]),
                test_points[6],
            ]

            const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
                test_points, 
                test_scalar_chunks,
                r.new_point_indices,
                r.cluster_start_indices,
            )
            expect(new_points.length).toEqual(expected_new_points.length)
            expect(new_scalar_chunks.length).toEqual(expected_new_scalar_chunks.length)
            for (let i = 0; i < expected_new_points.length; i ++) {
                expect(new_points[i]).toEqual(expected_new_points[i])
                expect(new_scalar_chunks[i]).toEqual(expected_new_scalar_chunks[i])
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

                    const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
                        points, 
                        scalar_chunks,
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

                    const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
                        points, 
                        scalar_chunks,
                        new_point_indices,
                        cluster_start_indices,
                    )
                }
            }
        })
    })
})
