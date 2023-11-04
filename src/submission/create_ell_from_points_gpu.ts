import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { decompose_scalars, bigIntPointToExtPointType } from './utils'
import { create_ell_gpu } from './create_ell'

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

export async function create_ell_sparse_matrices_from_points_gpu_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const num_threads = 8
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const ell_sms = await create_ell_sparse_matrices_from_points_gpu(points, scalars, num_threads)
    console.log(ell_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

const fieldMath = new FieldMath()

export async function create_ell_sparse_matrices_from_points_gpu(
    points: ExtPointType[],
    scalars: bigint[],
    num_threads: number,
): Promise<ELLSparseMatrix[]> {
    const ell_sms: ELLSparseMatrix[] = []
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(points.length % num_threads === 0)
    assert(points.length === scalars.length)

    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    const ell_sms_serial: ELLSparseMatrix[] = []
    for (const scalar_chunks of decomposed_scalars) {
        const ell_sm = create_ell_gpu(
            points,
            scalar_chunks,
            num_threads,
        )
        ell_sms_serial.push(ell_sm)
    }

    return ell_sms
}

