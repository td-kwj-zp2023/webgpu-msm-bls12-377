{{> bigint_struct }}

struct Point {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}

// Already defined in montgomery_product.template.wgsl
// const WORD_SIZE = {{ word_size }}u;
// const NUM_WORDS = {{ num_words }}u; 

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;
@group(0) @binding(1)
var<storage, read_write> output: array<Point>;

fn add_points(p1: Point, p2: Point) -> Point {
    // This is add-2008-hwcd-4
    // https://eprint.iacr.org/2008/522.pdf section 3.2, p7 (8M)
    // From http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
    
    // Operation counts
    // montgomery_product: 10 (2 of which are *2 - can this be further
    // optimised? The paper counts this as 8M)
    // fr_add: 4
    // fr_sub: 4

    var p1x = p1.x;
    var p1y = p1.y;
    var p2x = p2.x;
    var p2y = p2.y;
    var y1_s_x2 = fr_sub(&p1y, &p1x);
    var y2_a_x2 = fr_add(&p2y, &p2x);
    var a = montgomery_product(&y1_s_x2, &y2_a_x2);

    var y1_a_x2 = fr_add(&p1y, &p1x);
    var y2_s_x2 = fr_sub(&p2y, &p2x);
    var b = montgomery_product(&y1_a_x2, &y2_s_x2);

    var f = fr_sub(&b, &a);

    var p1z = p1.z;
    var p2t = p2.t;
    var z1_m_r2 = fr_double(&p1z);
    var c = montgomery_product(&z1_m_r2, &p2t);

    var p1t = p1.t;
    var p2z = p2.z;
    var t1_m_r2 = fr_double(&p1t);
    var d = montgomery_product(&t1_m_r2, &p2z);

    var e = fr_add(&d, &c);
    var g = fr_add(&b, &a);
    var h = fr_sub(&d, &c);
    var x3 = montgomery_product(&e, &f);
    var y3 = montgomery_product(&g, &h);
    var t3 = montgomery_product(&e, &h);
    var z3 = montgomery_product(&f, &g);

    return Point(x3, y3, t3, z3);
}

{{> montgomery_product_funcs }}

fn fr_double(a: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    bigint_double(a, &res);
    return fr_reduce(&res);
}

fn fr_add(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    bigint_add(a, b, &res);
    return fr_reduce(&res);
}

fn fr_reduce(a: ptr<function, BigInt>) -> BigInt {
    var res: BigInt;
    var p: BigInt = get_p();
    var underflow = bigint_sub(a, &p, &res);
    if (underflow == 1u) {
        return *a;
    }

    return res;
}

fn fr_sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;

    // if a > b: return a - b 
    // else:     return p - (b - a)

    var c = bigint_gt(a, b);
    if (c == 0u) { // a < b
        var r: BigInt;
        bigint_sub(b, a, &r);
        var p = get_p();
        bigint_sub(&p, &r, &res);
        return res;
    } else { // a > b
        bigint_sub(a, b, &res);
        return res;
    }
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var a = points[global_id.x];
    var b = points[global_id.x + 1u];
    var c = add_points(a, b);
    for (var i = 1u; i < {{ cost }}; i ++) {
        c = add_points(c, a);
    }
    output[global_id.x] = c;
}
