fn fr_double(a: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    bigint_double(a, &res);
    return fr_reduce(&res);
}

fn double_point(p1: Point) -> Point {
    var p1x = p1.x;
    var p1y = p1.y;
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

/*
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

    // Unfortunately this branch slows down the Tint compiler
    if (b_eq_a == 1u) {
        // The dedicated addition formula in add-2008-hwcd-4 assumes that a and
        // b are distinct, so double the point instead if a == b. 
        return double_point(p1);
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
*/

fn add_points(p1: Point, p2: Point) -> Point {
    // This is add-2008-hwcd
    // https://eprint.iacr.org/2008/522.pdf section 3.1, p5 (9M + 2D)
    // https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd
    // Operation counts
    // montgomery_product: 10
    // fr_add: 3
    // fr_sub: 5

    var p1x = p1.x;
    var p2x = p2.x;
    var a = montgomery_product(&p1x, &p2x);

    var p1y = p1.y;
    var p2y = p2.y;
    var b = montgomery_product(&p1y, &p2y);

    var p1t = p1.t;
    var p2t = p2.t;
    var t2 = montgomery_product(&p1t, &p2t);

    var EDWARDS_D = get_edwards_d();
    var c = montgomery_product(&EDWARDS_D, &t2);

    var p1z = p1.z;
    var p2z = p2.z;
    var d = montgomery_product(&p1z, &p2z);

    var xpy = fr_add(&p1x, &p1y);
    var xpy2 = fr_add(&p2x, &p2y);
    var e = montgomery_product(&xpy, &xpy2);
    e = fr_sub(&e, &a);
    e = fr_sub(&e, &b);

    var f = fr_sub(&d, &c);
    var g = fr_add(&d, &c);

    var p = get_p();
    var a_neg = fr_sub(&p, &a);

    var h = fr_sub(&b, &a_neg);
    var added_x = montgomery_product(&e, &f);
    var added_y = montgomery_product(&g, &h);
    var added_t = montgomery_product(&e, &h);
    var added_z = montgomery_product(&f, &g);

    return Point(added_x, added_y, added_t, added_z);
}

fn get_edwards_d() -> BigInt {
    var d: BigInt;
    d.limbs[0] = 760u;
    d.limbs[1] = 8111u;
    d.limbs[2] = 8191u;
    d.limbs[3] = 2047u;
    d.limbs[4] = 2879u;
    d.limbs[5] = 7826u;
    d.limbs[6] = 8149u;
    d.limbs[7] = 3887u;
    d.limbs[8] = 7498u;
    d.limbs[9] = 489u;
    d.limbs[10] = 5641u;
    d.limbs[11] = 7817u;
    d.limbs[12] = 1758u;
    d.limbs[13] = 6342u;
    d.limbs[14] = 5673u;
    d.limbs[15] = 2217u;
    d.limbs[16] = 2688u;
    d.limbs[17] = 7853u;
    d.limbs[18] = 7621u;
    d.limbs[19] = 20u;
    return d;
}
