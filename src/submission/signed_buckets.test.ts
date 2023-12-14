import { to_words_le, genRandomFieldElement } from './utils'

const to_signed_slices = (
    scalar: bigint,
    k: number, // The number of slices
    c: number, // The window size
) => {
    const l = 2 ** c

    const slices = to_words_le(scalar, k, c)
    const signed_slices: number[] = Array(slices.length).fill(0)

    let carry = 0
    for (let i = 0; i < k; i ++) {
        signed_slices[i] = slices[i] + carry
        if (signed_slices[i] >= l / 2) {
            signed_slices[i] = (l - signed_slices[i]) * -1
            carry = 1
        } else {
            carry = 0
        }
    }
    if (carry === 1) {
        console.error('final carry is 1', scalar)
    }
    return signed_slices
}

const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

describe('signed buckets', () => {
    // From https://hackmd.io/@drouyang/signed-bucket-index
    it('decompose scalars into signed chunks', () => {
        const num_tests = 102400
        for (let i = 0; i < num_tests; i ++) {
            const scalar = genRandomFieldElement(p)

            // The number of slices
            const k = 16

            // The window size
            const c = 16

            const signed_slices: number[] = to_signed_slices(scalar, k, c)

            // Sanity check
            let result = BigInt(0)
            for (let i = 0; i < signed_slices.length; i ++) {
                const b = BigInt(2) ** BigInt(i * c)
                result += BigInt(signed_slices[i]) * b
            }
            expect(result).toEqual(scalar)
        }
    })
})
