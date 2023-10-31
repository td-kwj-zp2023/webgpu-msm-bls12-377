import { to_words_le } from './utils'
import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { create_ell } from './create_ell'
import assert from 'assert'

// e.g. if the scalars converted to limbs = [
//    [limb_a, limb_b],
//    [limb_c, limb_d]
// ]
// return: [
//    [limb_a, limb_c],
//    [limb_b, limb_d]
// ]
const decompose_scalars = (
    scalars: bigint[],
    num_words: number,
    word_size: number,
): number[][] => {
    const as_limbs: number[][] = []
    for (const scalar of scalars) {
        const limbs = to_words_le(scalar, num_words, word_size)
        as_limbs.push(Array.from(limbs))
    }
    const result: number[][] = []
    for (let i = 0; i < num_words; i ++) {
        const t = as_limbs.map((limbs) => limbs[i])
        result.push(t)
    }
    return result
}

export async function create_ell_sparse_matrices_from_points_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const num_threads = 8
    const ell_sms = await create_ell_sparse_matrices_from_points(baseAffinePoints, scalars, num_threads)
    return { x: BigInt(0), y: BigInt(1) }
}

import { spawn, Thread, Worker } from 'threads'
const fieldMath = new FieldMath()

export async function create_ell_sparse_matrices_from_points(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    num_threads: number,
): Promise<ELLSparseMatrix[]> {
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(baseAffinePoints.length % num_threads === 0)
    assert(baseAffinePoints.length === scalars.length)

    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    const ell_sms: ELLSparseMatrix[] = []

    const web_worker = async (
        baseAffinePoints: BigIntPoint[],
        scalar_chunks: number[],
        num_threads: number,
    ) => {
        const worker = await spawn(new Worker('./createEllWorker.js'))
        const result = await worker(
            baseAffinePoints,
            scalar_chunks,
            num_threads,
        )
        await Thread.terminate(worker)
        return result
    }

    const start_webworkers = Date.now()
    const h = navigator.hardwareConcurrency
    const total_runs = decomposed_scalars.length
    const num_parallel_runs = Math.ceil(total_runs / h)
    for (let i = 0; i < num_parallel_runs; i ++) {
        const promises = []
        for (let j = 0; j < h; j ++) {
            const run_idx = i * h + j
            if (run_idx === total_runs) {
                break
            }
            promises.push(web_worker(baseAffinePoints, decomposed_scalars[run_idx], num_threads))
        }
        const results = await Promise.all(promises)
        for (const r of results) {
            ell_sms.push(r)
        }
    }
    const elapsed_webworkers = Date.now() - start_webworkers
    console.log(`Webworkers took ${elapsed_webworkers}ms`)

    const start_cpu = Date.now()
    const ell_sms_cpu: ELLSparseMatrix[] = []

    // For each set of decomposed scalars (e.g. the 0th chunks of each scalar,
    // the 1th chunks, etc, generate an ELL sparse matrix
    for (const scalar_chunks of decomposed_scalars) {
        const ell_sm = create_ell(
            baseAffinePoints,
            scalar_chunks,
            num_threads,
        )
        ell_sms_cpu.push(ell_sm)
    }
    const elapsed_cpu = Date.now() - start_cpu
    console.log(`CPU took ${elapsed_cpu}ms`)

    //console.log(ell_sms)
    //console.log(ell_sms_cpu)
    return ell_sms
}

