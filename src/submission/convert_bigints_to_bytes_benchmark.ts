import { BigIntPoint } from "../reference/types"
import { bigints_to_u8_for_gpu } from './utils'

const num_words = 20
const word_size = 13

export const convert_bigints_to_bytes_benchmark = async(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const start = Date.now()
    bigints_to_u8_for_gpu(scalars, num_words, word_size)
    const elapsed = Date.now() - start
    console.log(`CPU (serial) took ${elapsed}ms to convert ${scalars.length} scalars`)

    return { x: BigInt(0), y: BigInt(0) }
}
