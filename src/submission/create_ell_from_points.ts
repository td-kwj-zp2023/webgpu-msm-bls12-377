import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { decompose_scalars, bigIntPointToExtPointType } from './utils'
import { create_ell, prep_for_sort_method, prep_for_cluster_method } from './create_ell'

import assert from 'assert'
import { spawn, Thread, Worker } from 'threads'
import { ExtPointType } from "@noble/curves/abstract/edwards";

const fieldMath = new FieldMath()
const num_threads = 16

export async function prep_serial_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    func: any,
) {
    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
        const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
        for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
            //const { new_point_indices, cluster_start_indices }  = func(
            func(
                scalar_chunks,
                thread_idx,
                num_threads,
            )
        }
    }
}

export async function prep_for_cluster_method_serial_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    await prep_serial_benchmark(baseAffinePoints, scalars, prep_for_cluster_method)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function prep_for_sort_method_serial_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    await prep_serial_benchmark(baseAffinePoints, scalars, prep_for_sort_method)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function prep_webworker_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    worker_js: string,
) {
    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    const web_worker = async (
        scalar_chunks: number[],
        num_threads: number,
    ) => {
        const worker = await spawn(new Worker(worker_js))
        const result = await worker(
            scalar_chunks,
            num_threads,
        )
        await Thread.terminate(worker)
        return result
    }

    const promises = []
    for (let scalar_chunk_idx = 0; scalar_chunk_idx < decomposed_scalars.length; scalar_chunk_idx ++) {
        const scalar_chunks = decomposed_scalars[scalar_chunk_idx]
        promises.push(web_worker(scalar_chunks, num_threads))
    }
    await Promise.all(promises)
}

export async function prep_for_cluster_method_webworkers_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    await prep_webworker_benchmark(baseAffinePoints, scalars, './prepForClusterMethodWorker.js')
    return { x: BigInt(0), y: BigInt(1) }
}

export async function prep_for_sort_method_webworkers_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    await prep_webworker_benchmark(baseAffinePoints, scalars, './prepForSortMethodWorker.js')
    return { x: BigInt(0), y: BigInt(1) }
}

export async function create_ell_sparse_matrices_from_points_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const ell_sms = await create_ell_sparse_matrices_from_points(points, scalars, num_threads)
    console.log(ell_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function create_ell_sparse_matrices_from_points(
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

    const ell_sms_serial: ELLSparseMatrix[] = []
    let ell_sms_webworkers: ELLSparseMatrix[] = []

    const web_worker = async (
        points: ExtPointType[],
        scalar_chunks: number[],
        num_threads: number,
    ) => {
        const worker = await spawn(new Worker('./createEllWorker.js'))
        const result = await worker(
            points,
            scalar_chunks,
            num_threads,
        )
        await Thread.terminate(worker)
        return result
    }

    if (true) {
        const start_webworkers = Date.now()
        const promises = []
        for (let i = 0; i < decomposed_scalars.length; i ++) {
            promises.push(web_worker(points, decomposed_scalars[i], num_threads))
        }
        ell_sms_webworkers = await Promise.all(promises)

        const elapsed_webworkers = Date.now() - start_webworkers
        console.log(`Webworkers took ${elapsed_webworkers}ms`)
    }

    if (true) {
        const start_serial = Date.now()

        // For each set of decomposed scalars (e.g. the 0th chunks of each scalar,
        // the 1th chunks, etc, generate an ELL sparse matrix
        for (const scalar_chunks of decomposed_scalars) {
            const ell_sm = create_ell(
                points,
                scalar_chunks,
                num_threads,
            )
            ell_sms_serial.push(ell_sm)
        }
        const elapsed_serial = Date.now() - start_serial
        console.log(`serial took ${elapsed_serial}ms`)
    }

    // Sanity check
    assert(ell_sms_serial.length === ell_sms_webworkers.length)
    for (let i = 0; i < ell_sms_serial.length; i ++) {
        for (let j = 0; j < ell_sms_serial[i].data.length; j ++) {
            assert(ell_sms_serial[i].data[j].length === ell_sms_serial[i].row_length[j])
            assert(ell_sms_serial[i].col_idx[j].length === ell_sms_serial[i].row_length[j])
            assert(ell_sms_serial[i].row_length[j] === ell_sms_webworkers[i].row_length[j])

            for (let k = 0; k < ell_sms_serial[i].data[j].length; k ++) {
                assert(ell_sms_serial[i].data[j][k].ex === ell_sms_webworkers[i].data[j][k].ex)
                assert(ell_sms_serial[i].data[j][k].ey === ell_sms_webworkers[i].data[j][k].ey)
                assert(ell_sms_serial[i].data[j][k].et === ell_sms_webworkers[i].data[j][k].et)
                assert(ell_sms_serial[i].data[j][k].ez === ell_sms_webworkers[i].data[j][k].ez)
                assert(ell_sms_serial[i].col_idx[j][k] === ell_sms_webworkers[i].col_idx[j][k])
            }
        }
    }

    return ell_sms_serial
}
