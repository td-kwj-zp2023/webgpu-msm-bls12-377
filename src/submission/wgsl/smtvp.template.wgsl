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
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;

@group(0) @binding(2)
var<storage, read> row_ptr: array<u32>;

@group(0) @binding(3)
var<storage, read> points: array<Point>;

@group(0) @binding(4)
var<storage, read_write> loop_index: u32;

fn get_r() -> BigInt {
    var r: BigInt;

    r.limbs[0] = 7973u;
    r.limbs[1] = 8191u;
    r.limbs[2] = 8191u;
    r.limbs[3] = 3839u;
    r.limbs[4] = 1584u;
    r.limbs[5] = 8078u;
    r.limbs[6] = 8191u;
    r.limbs[7] = 129u;
    r.limbs[8] = 3124u;
    r.limbs[9] = 601u;
    r.limbs[10] = 7094u;
    r.limbs[11] = 6328u;
    r.limbs[12] = 4209u;
    r.limbs[13] = 259u;
    r.limbs[14] = 3351u;
    r.limbs[15] = 4579u;
    r.limbs[16] = 7118u;
    r.limbs[17] = 144u;
    r.limbs[18] = 6162u;
    r.limbs[19] = 14u;

    return r;
}

fn add_points(p1: Point, p2: Point) -> Point {
    // This is add-2008-hwcd-4
    // https://eprint.iacr.org/2008/522.pdf section 3.2, p7 (8M)
    // http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
    
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

    var b_eq_a = 1u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        if (a.limbs[i] != b.limbs[i]) {
            b_eq_a = 0u;
            break;
        }
    }

    if (b_eq_a == 1u) {
        // The dedicated addition formula in add-2008-hwcd-4 assumes that a and
        // b are distinct, so double the point instead if a == b. 

        var a = montgomery_product(&p1x, &p1x);
        var b = montgomery_product(&p1y, &p1y);
        var p1z = p1.z;
        var z1_m_z1 = montgomery_product(&p1z, &p1z);
        var c = fr_add(&z1_m_z1, &z1_m_z1);
        var p = get_p();
        var d = fr_sub(&p, &a);
        var x1_m_y1 = fr_add(&p1x, &p1y);
        var x1y1_m_x1y1 = montgomery_product(&x1_m_y1, &x1_m_y1);
        var x1y1_m_x1y1_s_a = fr_sub(&x1y1_m_x1y1, &a);
        var e = fr_sub(&x1y1_m_x1y1_s_a, &b);
        var g = fr_add(&d, &b);
        var f = fr_sub(&g, &c);
        var h = fr_sub(&d, &b);
        var x3 = montgomery_product(&e, &f);
        var y3 = montgomery_product(&g, &h);
        var t3 = montgomery_product(&e, &h);
        var z3 = montgomery_product(&f, &g);
        return Point(x3, y3, t3, z3);
    }

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
    // During the first pass, store the point at infinity in the output buffer.
    if (loop_index == 0u) {
        for (var i = 0u; i < arrayLength(&output); i ++) {
            var y: BigInt = get_r();
            var z: BigInt = y;

            var inf: Point;
            inf.y = y;
            inf.z = z;
            output[global_id.x + i] = inf;
        }
    }

    // Perform SMTVP on an array of 1s
    let i = loop_index;
    let row_start = row_ptr[global_id.x + i];
    let row_end = row_ptr[global_id.x + i + 1];
    for (var j = row_start; j < row_end; j ++) {
        // Note that points[global_id.x + j] is not multiplied by anything
        // since the vector is an array of 1
        var temp = points[global_id.x + j];
        var col = col_idx[global_id.x + j];
        var ycol = output[global_id.x + col];
        var res = add_points(ycol, temp);

        // Store the result
        output[global_id.x + col] = res;
    }
}
