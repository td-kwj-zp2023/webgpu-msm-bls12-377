import { to_words_le } from './utils'
import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { decompose_scalars, bigIntPointToExtPointType } from './utils'
import { create_ell, gen_add_to } from './create_ell'

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

export async function create_ell_sparse_matrices_from_points_gpu_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const num_threads = 8
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const ell_sms = await create_ell_sparse_matrices_from_points_gpu(points, scalars, num_threads)
    //console.log(ell_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

import { spawn, Thread, Worker } from 'threads'
const fieldMath = new FieldMath()

export async function create_ell_sparse_matrices_from_points_gpu(
    points: ExtPointType[],
    scalars: bigint[],
    num_threads: number,
): Promise<ELLSparseMatrix[]> {
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(points.length % num_threads === 0)
    assert(points.length === scalars.length)

    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    // Benchmark gen_add_to
    const start_gat = Date.now()
    for (const scalar_chunks of decomposed_scalars) {
        const num_cols = scalar_chunks.length / num_threads
        
        for (let i = 0; i < num_threads; i ++) {
            // Take each num_thread-th chunk only (each row)
            const chunks: number[] = []
            for (let j = 0; j < num_cols; j ++) {
                const idx = i * num_cols + j
                const c = scalar_chunks[idx]
                chunks.push(c)
            }

            // Pre-aggregate points per row
            const { add_to, new_chunks } = gen_add_to(chunks)
        }
    }
    const elapsed_gat = Date.now() - start_gat
    console.log(`gen_add_to elapsed: ${elapsed_gat}ms`)

    // Benchmark the sorting method
    const start_sort = Date.now()
    for (const scalar_chunks of decomposed_scalars) {
        const num_cols = scalar_chunks.length / num_threads
        
        for (let i = 0; i < num_threads; i ++) {
            const pt_and_chunks = []
            for (let j = 0; j < num_cols; j ++) {
                const idx = i * num_cols + j
                const c = scalar_chunks[idx]
                pt_and_chunks.push([idx, c])
            }

            // Sort by chunk
            pt_and_chunks.sort((a: number[], b: number[]) => {
                if (a[1] > b[1]) { return 1 }
                else if (a[1] < b[1]) { return -1 }
                return 0
            })

            // Array containing starting indices of each cluster of unique
            // chunks
            const cluster_start_indices = [0]
            let prev_chunk = pt_and_chunks[0][1]
            for (let k = 1; k < pt_and_chunks.length; k ++) {
                if (prev_chunk !== pt_and_chunks[k][1]) {
                    cluster_start_indices.push(k)
                }
                prev_chunk = pt_and_chunks[k][1]
            }
            // TODO: this is going to be fed to the GPU
        }
    }
    const elapsed_sort = Date.now() - start_sort
    console.log(`sort method elapsed: ${elapsed_sort}ms`)

    const ell_sms: ELLSparseMatrix[] = []
    // TODO
    return ell_sms
}

