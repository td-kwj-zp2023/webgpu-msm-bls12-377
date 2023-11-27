/*
 * Adapted from https://github.com/ingonyama-zk/modular_multiplication
 */

const W_MASK = {{ w_mask }}u;
const SLACK = {{ slack }}u;

fn get_m() -> BigInt {
    var m: BigInt;
{{{ m_limbs }}}
    return m;
}

fn get_p_wide() -> BigIntWide {
    var p: BigIntWide;
{{{ p_limbs }}}
    return p;
}

fn machine_multiply(a: u32, b: u32) -> vec2<u32> {
    let ab = a * b;
    let hi = ab >> WORD_SIZE;
    let lo = ab & MASK;
    return vec2(lo, hi);
}

fn machine_two_digit_add(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    var carry = 0u;
    var res = vec3(0u, 0u, 0u);
    for (var i = 0u; i < 2u; i ++) {
        let sum = a[i] + b[i] + carry;
        res[i] = sum & MASK;
        carry = sum >> WORD_SIZE;
    }
    res[2] = carry;
    return res;
}

/*
 * Bitshift to the left. The shift value must be greater than the word size.
 */
fn mp_shifter_left(a: ptr<function, BigIntWide>, shift: u32) -> BigIntWide {
    var res: BigIntWide;
    var carry = 0u;
    let x = shift - WORD_SIZE;
    for (var i = 1u; i < NUM_WORDS * 2u; i ++) {
        res.limbs[i] = (((*a).limbs[i - 1] << x) & W_MASK) + carry;
        carry = (*a).limbs[i - 1] >> (WORD_SIZE - x);
    }
    return res;
}

fn mp_shifter_right(a: ptr<function, BigIntMediumWide>, shift: u32) -> BigInt {
    var res: BigInt;
    var borrow = 0u;
    let borrow_shift = WORD_SIZE - shift;
    let two_w = 1u << WORD_SIZE;
    for (var idx = 0u; idx < NUM_WORDS; idx ++) {
        let i = NUM_WORDS - idx - 1u;
        let new_borrow = (*a).limbs[i] << borrow_shift;
        res.limbs[i] = (((*a).limbs[i] >> shift) | borrow) % two_w;
        borrow = new_borrow;
    }
    return res;
}

fn mp_msb_multiply(a: ptr<function, BigIntWide>, b: ptr<function, BigInt>) -> BigInt {
    var c: array<u32, NUM_WORDS * 2 + 1>;
    for (var l = NUM_WORDS - 1u; l < NUM_WORDS * 2u - 1u; l ++) {
        let i_min = l - (NUM_WORDS - 1u);
        for (var i = i_min; i < NUM_WORDS; i ++) {
            let mult_res = machine_multiply((*a).limbs[NUM_WORDS + i], (*b).limbs[l-i]);
            let add_res = machine_two_digit_add(mult_res, vec2(c[l], c[l+1]));
            c[l] = add_res[0];
            c[l + 1] = add_res[1];
            c[l + 2] = c[l + 2] + add_res[2];
        }
    }

    var result: BigInt;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        result.limbs[i] = c[NUM_WORDS + i];
    }
    return result;
}

fn mp_lsb_multiply(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigIntMediumWide {
    var c: array<u32, NUM_WORDS * 2 + 1>;
    for (var l = 0u; l < NUM_WORDS; l ++) {
        let i_min = max(0i, i32(l) - (i32(NUM_WORDS) - 1i));
        let i_max = min(i32(l), (i32(NUM_WORDS) - 1i)) + 1i;  // + 1 for inclusive
        for (var i = i_min; i < i_max; i ++) {
            let mult_res = machine_multiply((*a).limbs[i], (*b).limbs[l - u32(i)]);
            let add_res = machine_two_digit_add(mult_res, vec2(c[l], c[l + 1]));
            c[l] = add_res[0];
            c[l + 1] = add_res[1];
            c[l + 2] = c[l + 2] + add_res[2];
        }
    }
    var result: BigIntMediumWide;
    for (var i = 0u; i < NUM_WORDS + 1u; i ++) {
        result.limbs[i] = c[i];
    }
    return result;
}

fn mp_adder(a: ptr<function, BigInt>, b: ptr<function, BigIntWide>) -> BigIntMediumWide {
    var c: BigIntMediumWide;
    var carry = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        let x = (*a).limbs[i] + (*b).limbs[NUM_WORDS + i] + carry;
        c.limbs[i] = x & MASK;
        carry = x >> WORD_SIZE;
    }
    return c;
}

fn mp_subtracter(a: ptr<function, BigIntWide>, b: ptr<function, BigIntMediumWide>) -> BigInt {
    var res: BigInt;
    var borrow = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        res.limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            res.limbs[i] += TWO_POW_WORD_SIZE;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return res;
}

fn mp_full_multiply(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigIntWide {
    var c: array<u32, NUM_WORDS * 2 + 1>;
    for (var l = 0u; l < NUM_WORDS * 2u - 1u; l ++) {
        let i_min = u32(max(0i, i32(l) - i32(NUM_WORDS - 1u)));
        let i_max = u32(min(l, NUM_WORDS - 1u) + 1u);  // + 1 for inclusive
        for (var i = i_min; i < i_max; i ++) {
            let mult_res = machine_multiply((*a).limbs[i], (*b).limbs[l - u32(i)]);
            let add_res = machine_two_digit_add(mult_res, vec2(c[l], c[l+1]));
            c[l] = add_res[0];
            c[l + 1] = add_res[1];
            c[l + 2] += add_res[2];
        }
    }
    var result: BigIntWide;
    for (var i = 0u; i < NUM_WORDS * 2u; i ++) {
        result.limbs[i] = c[i];
    }
    return result;
}

fn mp_subtract_red(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    var res = *a;
    var r: BigInt;
    while (bigint_gt(&res, b) == 1u) {
        bigint_sub(&res, b, &r);
        res = r;
    }
    return res;
}

fn field_mul(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    var ab = mp_full_multiply(a, b);
    let z = {{ z }}u;

    // AB msb extraction (+ shift)
    var ab_shift = mp_shifter_left(&ab, z * 2u);

    // L estimation
    var m = get_m();

    var l = mp_msb_multiply(&ab_shift, &m); // calculate l estimator (MSB multiply)
    var l_add_ab_msb = mp_adder(&l, &ab_shift);

    l = mp_shifter_right(&l_add_ab_msb, z);
    var p = get_p();

    // LS calculation
    var ls_mw: BigIntMediumWide = mp_lsb_multiply(&l, &p);

    var result = mp_subtracter(&ab, &ls_mw);
    return mp_subtract_red(&result, &p);
}
