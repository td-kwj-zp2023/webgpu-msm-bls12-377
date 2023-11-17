import assert from 'assert'
import * as wasm from 'decompose-scalars'
import { BigIntPoint } from "../reference/types"
import {
    to_words_le,
    decompose_scalars,
    compute_misc_params,
} from './utils'

export const decompose_scalars_ts_benchmark = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

    //console.log(Array.from(to_words_le(BigInt(p), 20, 13)))//.map((x) => BigInt(x).toString(16)))
    // [1, 0, 0, 768, 4257, 0, 0, 8154, 2678, 2765, 3072, 6255, 4581, 6694, 6530, 5290, 6700, 2804, 2777, 37]

    console.log('Typescript benchmarks:')
    //for (let word_size = 13; word_size < 14; word_size ++) {
    for (let word_size = 8; word_size < 20; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start = Date.now()
        decompose_scalars(scalars, num_words, word_size)
        const elapsed = Date.now() - start
        console.log(`decompose_scalars() with ${word_size}-bit windows took ${elapsed}ms`)
    }
    console.log()

    console.log('WASM benchmarks:')
    //for (let word_size = 13; word_size < 14; word_size ++) {
    for (let word_size = 8; word_size < 20; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start_wasm = Date.now()
        wasm.decompose_scalars(scalars, num_words, word_size).get_result()
        const elapsed_wasm = Date.now() - start_wasm
        console.log(`WASM with ${word_size}-bit windows took ${elapsed_wasm}ms`)
    }

    const num_words = 20
    const word_size = 13
    const ts_r = decompose_scalars(scalars, num_words, word_size).flat()
    const wasm_r = wasm.decompose_scalars(scalars, num_words, word_size).get_result()
    assert(ts_r.toString() === wasm_r.toString())
    console.log('ok')
    //debugger

    //console.log('GPU:')
    // Convert scalars to bytes

    return { x: BigInt(0), y: BigInt(0) }
}
