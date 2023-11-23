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
    const hi = ab >> word_size
    const lo = ab & mask
    return [lo, hi]
}

export const mp_shifter_left = (
    a_words: Uint16Array,
    shift: number,
    num_words: number,
    word_size: number,
) => {
    const res = Array(num_words + 1).fill(0)
    const w_mask = (1 << word_size) - 1
    for (let i = 0; i < num_words; i ++) {
        const s = a_words[i] << shift
        res[i] = res[i] | (s & w_mask)
        res[i + 1] = s >> word_size
    }
    return res.slice(0, num_words)
}

export const mp_shifter_right = (
    a_words: Uint16Array,
    shift: number,
    num_words: number,
    word_size: number,
) => {
    const res = Array(num_words).fill(0)
    const two_w = 2 ** word_size
    let borrow = 0
    const borrow_shift = word_size - shift
    for (let idx = 0; idx < num_words; idx ++) {
        const i = word_size - idx - 1
        const new_borrow = a_words[i] << borrow_shift
        res[i] = ((a_words[i] >> shift) | borrow) % two_w
        borrow = new_borrow
    }
    return res
}

export const machine_two_digit_add = (
    a: number[],
    b: number[],
    word_size: number,
) => {
    const mask = (2 ** word_size) - 1
    let carry = 0
    const res = [0, 0, 0]
    for (let i = 0; i < 2; i ++) {
        const sum = a[i] + b[i] + carry
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
) => {
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
    return c.slice(num_words, 2 * num_words)
}

export const mp_lsb_multiply = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
    return_extra_digit: boolean,
) => {
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
        return c.slice(0, num_words + 1)
    } else {
        return c.slice(0, num_words)
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
) => {
    const w = word_size ** word_size
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
    return res
}
    /*
fn bigint_sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>, res: ptr<function, BigInt>) -> u32 {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < NUM_WORDS; i = i + 1u) {
        (*res).limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            (*res).limbs[i] += TWO_POW_WORD_SIZE;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
    */

export const mp_full_multiply = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    num_words: number,
    word_size: number,
) => {
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
    return c.slice(0, num_words * 2)
}

export const barrett_domb_mul = (
    a_words: Uint16Array,
    b_words: Uint16Array,
    p_words: Uint16Array,
): Uint16Array => {
    const result = new Uint16Array()
    return result
}
