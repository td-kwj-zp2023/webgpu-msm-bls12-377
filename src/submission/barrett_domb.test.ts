import {
    to_words_le,
    genRandomFieldElement,
} from './utils'
import {
    calc_m,
    mp_adder,
    mp_subtracter,
    mp_shifter_left,
    mp_msb_multiply,
    mp_lsb_multiply,
    mp_shifter_right,
    barrett_domb_mul,
    machine_multiply,
    mp_full_multiply,
    machine_two_digit_add,
    mp_lsb_extra_diagonal,
} from './barrett_domb'

describe('Barrett-Domb ', () => {
    it('calc_m', () => {
        const p = BigInt('0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001')
        const word_size = 16
        const m = calc_m(p, word_size)
        expect(m).toEqual(BigInt('153139381818454018477577869068761289266858026142902538442762978823351945372822'))
    })

    it('mp_lsb_extra_diagonal', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const b = BigInt('0x2222222222222222222222222222222222222222222222222222222222222222')
        const b_words = to_words_le(b, num_words, word_size)
        const result = mp_lsb_extra_diagonal(a_words, b_words, num_words, word_size)
        expect(result).toEqual(8158)
    })

    it('mp_msb_multiply', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const b = BigInt('0x2222222222222222222222222222222222222222222222222222222222222222')
        const b_words = to_words_le(b, num_words, word_size)

        const result = mp_msb_multiply(a_words, b_words, num_words, word_size)
        expect(result.toString()).toEqual(
            [18063, 48642, 13689, 44273, 9320, 39904, 4951, 35535, 582, 31166, 61749, 26796, 57380, 22427, 53011, 1162].toString())
    })

    it('mp_lsb_multiply', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const b = BigInt('0x2222222222222222222222222222222222222222222222222222222222222222')
        const b_words = to_words_le(b, num_words, word_size)

        const result = mp_lsb_multiply(a_words, b_words, num_words, word_size, true)
        expect(result.toString()).toEqual(
            [3782, 38739, 8155, 43108, 12524, 47477, 16893, 51846, 21262, 56215, 25631, 60584, 30000, 64953, 34369, 20682, 9905].toString())
    })

    it('mp_adder', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const b = BigInt('0x2222222222222222222222222222222222222222222222222222222222222222')
        const b_words = to_words_le(b, num_words, word_size)

        const result = mp_adder(a_words, b_words, num_words, word_size)
        const expected = to_words_le(a + b, num_words + 1, word_size)
        expect(result.toString()).toEqual(expected.toString())
    })

    it('mp_subtracter', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const b = BigInt('0x1111111111111111111111111111111111111111111111111111111111111111')
        const b_words = to_words_le(b, num_words, word_size)

        const result = mp_subtracter(a_words, b_words, num_words, word_size)
        const expected = to_words_le(a - b, num_words, word_size)
        expect(result.toString()).toEqual(expected.toString())
    })

    it('mp_shifter_left', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x2211111111111111111111111111111111111111111111111111111111111133')
        const a_words = to_words_le(a, num_words, word_size)
        const shift = 3
        const expected = to_words_le(a << BigInt(shift), num_words, word_size)
        const result = mp_shifter_left(a_words, shift, num_words, word_size)
        expect(result.toString()).toEqual(expected.toString())
    })

    it('mp_shifter_right', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x1111111111111111111111111111111111111111111111111111111111111111')
        const a_words = to_words_le(a, num_words, word_size)
        const shift = 3
        const expected = to_words_le(a >> BigInt(shift), num_words, word_size)
        const result = mp_shifter_right(a_words, shift, num_words, word_size)
        expect(result.toString()).toEqual(expected.toString())
    })

    it('mp_full_multiply', () => {
        const num_words = 16
        const word_size = 16
        const a = BigInt('0x1111111111111111111111111111111111111111111111111111111111111111')
        const a_words = to_words_le(a, num_words, word_size)

        const b = BigInt('0x2222222222222222222222222222222222222222222222222222222222222222')
        const b_words = to_words_le(b, num_words, word_size)

        const result = mp_full_multiply(a_words, b_words, num_words, word_size)

        const expected = to_words_le(a * b, num_words * 2, word_size)

        expect(result.toString()).toEqual(expected.toString())
    })

    it('machine_multiply', () => {
        //const a = 0xff
        //const b = 0xfab
        const a = 44273
        const b = 53641
        const res = machine_multiply(a, b, 16)
        expect(res.toString()).toEqual([19961, 36237].toString())
    })

    it('machine_two_digit_add', () => {
        const a = [25512, 7628]
        const b = [55494, 14279]
        const expected = [15470, 21908, 0]

        //const a = [0xfff0, 0xfffe]
        //const b = [0xfff1, 0xfff2]
        const res = machine_two_digit_add(a, b, 16)
        expect(res.toString()).toEqual(expected.toString())
    })

    it('barrett_domb_mul', () => {
        const p = BigInt('0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001')
        const n = p.toString(2).length
        const word_size = 13
        const num_words = Math.ceil(n / word_size)

        const p_words = to_words_le(p, num_words, word_size)

        const a = BigInt('12606796758224846727326035948803889824738128609730484894316100840265045196027')
        const b = BigInt('14276552610056165753848820256553331055663673083569154091093918058913504134283')

        const a_words = to_words_le(a, num_words, word_size)

        const b_words = to_words_le(b, num_words, word_size)

        const m = calc_m(p, word_size)
        const m_words = to_words_le(m, num_words, word_size)

        const result = barrett_domb_mul(a_words, b_words, p_words, p.toString(2).length, m_words, num_words, word_size)

        const expected = a * b % p
        const expected_words = to_words_le(expected, num_words, word_size)

        expect(result.toString()).toEqual(expected_words.toString())
    })

    it('barrett_domb_mul with random inputs', () => {
        const num_runs = 1000
        const word_size = 13

        const p = BigInt('0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001')
        const n = p.toString(2).length

        for (let i = 0; i < num_runs; i ++) {
            const num_words = Math.ceil(n / word_size)

            const p_words = to_words_le(p, num_words, word_size)

            const a = genRandomFieldElement(p)
            const a_words = to_words_le(a, num_words, word_size)

            const b = genRandomFieldElement(p)
            const b_words = to_words_le(b, num_words, word_size)

            //console.log(a, b)

            const m = calc_m(p, word_size)
            const m_words = to_words_le(m, num_words, word_size)

            const result = barrett_domb_mul(a_words, b_words, p_words, p.toString(2).length, m_words, num_words, word_size)
            const expected = a * b % p
            const expected_words = to_words_le(expected, num_words, word_size)

            expect(result.toString()).toEqual(expected_words.toString())
        }
    })
})
