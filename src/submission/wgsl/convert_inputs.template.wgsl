{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}
{{> montgomery_product_funcs }}

const W_mask = {{ w_mask }}u;
const SLACK = {{ slack }}u;

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

fn get_mu() -> BigInt {
    var mu: BigInt;
{{{ mu_limbs }}}
    return mu;
}

fn get_p_wide() -> BigIntWide {
    var p: BigIntWide;
{{{ p_limbs }}}
    return p;
}

fn mul(a: BigInt, b: BigInt) -> BigIntWide {
    var res: BigIntWide;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        for (var j = 0u; j < NUM_WORDS; j = j + 1u) {
            let c = a.limbs[i] * b.limbs[j];
            res.limbs[i+j] += c & W_mask;
            res.limbs[i+j+1] += c >> WORD_SIZE;
        }   
    }

    // start from 0 and carry the extra over to the next index
    for (var i = 0u; i < 2 * NUM_WORDS - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> WORD_SIZE;
        res.limbs[i] = res.limbs[i] & W_mask;
    }
    return res;
}

fn sub_512(a: BigIntWide, b: BigIntWide, res: ptr<function, BigIntWide>) -> u32 {
    var borrow = 0u;
    for (var i = 0u; i < 2u * NUM_WORDS; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1u;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn get_higher_with_slack(a: BigIntWide) -> BigInt {
    var out: BigInt;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        out.limbs[i] = ((a.limbs[i + NUM_WORDS] << SLACK) + (a.limbs[i + NUM_WORDS - 1] >> (WORD_SIZE - SLACK))) & W_mask;
    }
    return out;
}

fn field_mul(a: BigInt, b: BigInt) -> BigInt {
    var xy: BigIntWide = mul(a, b);
    var xy_hi: BigInt = get_higher_with_slack(xy);
    var l: BigIntWide = mul(xy_hi, get_mu());
    var l_hi: BigInt = get_higher_with_slack(l);
    var lp: BigIntWide = mul(l_hi, get_p());
    var r_wide: BigIntWide;
    sub_512(xy, lp, &r_wide);

    var r_wide_reduced: BigIntWide;
    var underflow = sub_512(r_wide, get_p_wide(), &r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BigInt;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return fr_reduce(&r);
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 

    let id = gidx * 8 + gidy;
    let pt = points[id];

    let r = get_r();

    var xr = field_mul(pt.x, r);
    var yr = field_mul(pt.y, r);
    let tr = montgomery_product(&xr, &yr);
    let z = r;
    
    let new_pt = Point(xr, yr, tr, z);
    points[id] = new_pt;
}
