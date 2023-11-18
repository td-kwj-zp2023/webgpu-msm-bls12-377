import { BigIntPoint } from "../../reference/types"

export const cuzk_gpu = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    return { x: BigInt(1), y: BigInt(0) }
}
