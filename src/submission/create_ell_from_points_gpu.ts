import { to_words_le } from './utils'
import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { decompose_scalars, bigIntPointToExtPointType } from './utils'
import { create_ell } from './create_ell'

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

const fieldMath = new FieldMath()

export async function create_ell_sparse_matrices_from_points_gpu(
    points: ExtPointType[],
    scalars: bigint[],
    num_threads: number,
): Promise<ELLSparseMatrix[]> {
    const ell_sms: ELLSparseMatrix[] = []
    // TODO
    return ell_sms
}

