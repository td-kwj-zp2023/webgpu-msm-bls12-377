import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { cuzk_gpu } from './cuzk/cuzk_gpu'
import { Curve } from './curves'

export const compute_msm = async (
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
    scalars: bigint[] | Uint32Array[]
): Promise<{x: bigint, y: bigint}> => {
    return await cuzk_gpu(
        baseAffinePoints,
        scalars,
        Curve.BLS12_377,
    )
};
