import {
    to_words_le,
    from_words_le,
    compute_misc_params,
} from './utils'

describe('utils', () => {
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const word_size = 13
    const num_words = 20

    const testData: [bigint, Uint16Array, number][] = [[p, new Uint16Array([1, 0, 0, 768, 4257, 0, 0, 8154, 2678, 2765, 3072, 6255, 4581, 6694, 6530, 5290, 6700, 2804, 2777, 37]), word_size]]

    describe('bigint to limbs and vice versa', () => {
        it.each(testData)('to_words_le', (val: bigint, expected: Uint16Array, word_size: number) => {
            const words = to_words_le(val, num_words, word_size)
            expect(words).toEqual(expected)
        })

        it.each(testData)('from_words_le', (expected: bigint, words: Uint16Array, word_size: number) => {
            const val = from_words_le(words, num_words, word_size)
            expect(val).toEqual(expected)
        })
    })

    describe('misc functions', () => {
        it('compute_misc_params', () => {
            const num_words = 20
            const r = BigInt(2) ** BigInt(num_words * word_size)
            const expected = { num_words, max_terms: 40, k: 65, nsafe: 32, n0: BigInt(8191), r }
            const misc = compute_misc_params(p, word_size)
            expect(misc).toEqual(expected)
        })
    })
})

export {}
