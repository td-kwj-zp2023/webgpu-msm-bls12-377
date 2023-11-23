/*
 * Adapted from https://github.com/ingonyama-zk/modular_multiplication
 */

export const calc_m = (
    p: bigint,
    word_size: number,
) => {
    const n = p.toString(2).length
    const k = Math.ceil(n / word_size)
    const z = k * word_size - n
    const t = 2 ** (2 * n + z)
    const m = BigInt(t) / p
    return m
}

export const machine_multiply = (
    a: number,
    b: number,
    word_size: number,
) => {
    const mask = (2 ** word_size) - 1
    const ab = a * b
    const hi = Number(BigInt(ab) >> BigInt(word_size))
    const lo = ab & mask
    return [lo, hi]
}

export const mp_shifter_left = (
    a_words: Uint16Array,
    shift: number,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const res = Array(num_words + 1).fill(0)
    const w_mask = (1 << word_size) - 1
    for (let i = 0; i < num_words; i ++) {
        const s = a_words[i] << shift
        res[i] = res[i] | (s & w_mask)
        res[i + 1] = s >> word_size
    }
    return new Uint16Array(res.slice(0, num_words))
}

export const mp_shifter_right = (
    a_words: Uint16Array,
    shift: number,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const res = Array(num_words).fill(0)
    const two_w = 2 ** word_size
    let borrow = 0
    const borrow_shift = word_size - shift
    for (let idx = 0; idx < num_words; idx ++) {
        const i = num_words - idx - 1
        const new_borrow = a_words[i] << borrow_shift
        res[i] = ((a_words[i] >> shift) | borrow) % two_w
        borrow = new_borrow
    }
    return new Uint16Array(res)
}

export const machine_two_digit_add = (
    a_words: number[],
    b_words: number[],
    word_size: number,
): number[] => {
    const mask = (2 ** word_size) - 1
    let carry = 0
    const res = [0, 0, 0]
    for (let i = 0; i < 2; i ++) {
        const sum = a_words[i] + b_words[i] + carry
        res[i] = sum & mask
        carry = sum >> word_size
    }
    res[2] = carry
    return res
}

export const mp_msb_multiply = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const c = Array(num_words * 2 + 1).fill(0)
    for (let l = num_words - 1; l < num_words * 2 - 2 + 1; l ++) {
        const i_min = l - (num_words - 1)
        const i_max = num_words - 1 + 1  // + 1 for inclusive
        for (let i = i_min; i < i_max; i ++) {
            const mult_res = machine_multiply(
                a_words[i], 
                b_words[l-i],
                word_size
            )
            const add_res = machine_two_digit_add(
                mult_res,
                [c[l], c[l+1]],
                word_size,
            )
            c[l] = add_res[0]
            c[l + 1] = add_res[1]
            c[l + 2] = c[l + 2] + add_res[2]
        }
    }
    return new Uint16Array(c.slice(num_words, 2 * num_words))
}

export const mp_lsb_multiply = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
    return_extra_digit: boolean,
): Uint16Array => {
    const c = Array(num_words * 2 + 1).fill(0)
    for (let l = 0; l < num_words; l ++) {
        const i_min = Math.max(0, l - (num_words - 1))
        const i_max = Math.min(l, num_words - 1) + 1  // + 1 for inclusive
        for (let i = i_min; i < i_max; i ++) {
            const mult_res = machine_multiply(
                a_words[i],
                b_words[l-i],
                word_size,
            )
            const add_res = machine_two_digit_add(
                mult_res,
                [c[l], c[l+1]],
                word_size,
            )
            c[l] = add_res[0]
            c[l + 1] = add_res[1]
            c[l+2] = c[l + 2] + add_res[2]
        }
    }

    if (return_extra_digit) {
        // Will be true in barrett_domb_mul, so the WGSL version should just
        // use this branch
        return new Uint16Array(c.slice(0, num_words + 1))
    } else {
        return new Uint16Array(c.slice(0, num_words))
    }
}

export const mp_adder = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
) => {
    const c = Array(num_words + 1).fill(0)
    const mask = (2 ** word_size) - 1
    let carry = 0
    for (let i = 0; i < num_words; i ++) {
        const x = a_words[i] + b_words[i] + carry
        c[i] = x & mask
        carry = x >> word_size
    }
    return c
}

export const mp_subtracter = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const res = Array(num_words).fill(0)
    const two_w = 2 ** word_size
    let borrow = 0
    for (let i = 0; i < num_words; i ++) {
        res[i] = a_words[i] - b_words[i] - borrow
        if (a_words[i] < (b_words[i] + borrow)) {
            res[i] += two_w
            borrow = 1
        } else {
            borrow = 0
        }
    }
    return new Uint16Array(res)
}

export const mp_full_multiply = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const c = new Uint16Array(num_words * 2 + 1)
    for (let l = 0; l < num_words * 2 - 2 + 1; l ++) {
        const i_min = Math.max(0, l - (num_words - 1))
        const i_max = Math.min(l, num_words - 1) + 1  // + 1 for inclusive
        for (let i = i_min; i < i_max; i ++) {
            const mult_res = machine_multiply(
                a_words[i],
                b_words[l-i],
                word_size,
            )
            const add_res = machine_two_digit_add(
                mult_res,
                [c[l], c[l+1]],
                word_size,
            )
            c[l] = add_res[0]
            c[l + 1] = add_res[1]
            c[l + 2] += add_res[2]
        }
    }
    return new Uint16Array(c.slice(0, num_words * 2))
}

export const mp_lsb_extra_diagonal = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
): number => {
    const c = Array(num_words * 2 + 1).fill(0)
    const l = num_words
    const i_min = Math.max(0, l - (num_words - 1))
    const i_max = Math.min(l, num_words - 1) + 1  // + 1 for inclusive
    for (let i = i_min; i < i_max; i ++) {
        const mult_res = machine_multiply(a_words[i], b_words[l - i], word_size)
        const add_res = machine_two_digit_add(Array.from(mult_res), [c[l], c[l + 1]], word_size)
        c[l] = add_res[0]
        c[l + 1] = add_res[1]
        c[l + 2] = c[l + 2] + add_res[2]
    }
    return c[l]
}

export const mp_gt = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
) => {
    for (let idx = 0; idx < num_words; idx ++) {
        const i = num_words - 1 - idx
        if (a_words[i] < b_words[i]) {
            return false
        } else {
            return true
        }
    }
    return false
}

export const mp_subtract_red = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
) => {
    let res = a_words
    while (mp_gt(res, b_words, num_words)) {
        res = mp_subtracter(res, b_words, num_words, word_size)
    }
    return res
}

export const barrett_domb_mul = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    p_words: Uint16Array,
    p_bitlen: number,
    m_words: Uint16Array,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const n = p_bitlen
    const k = Math.ceil(n / word_size)

    // multiply and break into LSB, MSB parts
    const ab = mp_full_multiply(a_words, b_words, word_size, k)
    const wk = word_size * k
    const z = wk - n

    // AB msb extraction (+ shift)
    const ab_shift = mp_shifter_left(ab, 2 * z, k * 2, word_size)

    const ab_msb = new Uint16Array(ab_shift.slice(k, k * 2))

    // L estimation
    let l = mp_msb_multiply(ab_msb, m_words, k, word_size) // calculate l estimator (MSB multiply)
    l = new Uint16Array(mp_adder(l, ab_msb, k, word_size).slice(0, k)) // Add another AB_msb because m[n] = 1
    l = mp_shifter_right(l, z, k, word_size)

    // LS calculation
    let ls = mp_lsb_multiply(l, p_words, word_size, k, true)

    // If needed, calculate extra diagonal.
    if (z < Math.log2(4 + k/(2**z))) {
        const lsb_mult_carry_extra = ls[k]
        const lsb_mult_extra = mp_lsb_extra_diagonal(l, p_words, word_size, k)
        ls[k] = lsb_mult_carry_extra + lsb_mult_extra
    } else {
        ls = new Uint16Array(Array.from(ls).slice(0, k))
    }

    let ab_lsb
    // adders and sub, not in multiprecision.
    if (z < Math.log2(4 + k / (2 ** z))) {
        ab_lsb = ab.slice(0, k + 1)
    } else {
        ab_lsb = ab.slice(0, k)
    }

    let result = mp_subtracter(ab_lsb, ls, num_words, word_size)

    result = mp_subtract_red(result, p_words, word_size, k)

    return result
}
