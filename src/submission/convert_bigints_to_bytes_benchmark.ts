import { BigIntPoint } from "../reference/types"
import { bigints_to_u8_for_gpu, bigints_to_16_bit_words_for_gpu } from './utils'
import assert from 'assert'

const num_words = 16
const word_size = 16

export const convert_bigints_to_bytes_benchmark = async(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const start = Date.now()

    const x_y_coords_bytes = bigints_to_u8_for_gpu(scalars, num_words, word_size)
    const x_y_coords_bytes_2 = bigints_to_16_bit_words_for_gpu(scalars)

    assert(x_y_coords_bytes.length == x_y_coords_bytes_2.length)
    for (let i = 0; i < x_y_coords_bytes.length; i++) {
        assert(x_y_coords_bytes[i] == x_y_coords_bytes_2[i])
    }

    console.log("passed assertion checks!")

    const elapsed = Date.now() - start
    console.log(`CPU (serial) took ${elapsed}ms to convert ${scalars.length} scalars`)

    return { x: BigInt(0), y: BigInt(0) }
}
