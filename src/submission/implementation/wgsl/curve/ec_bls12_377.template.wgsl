fn is_zero(coord: BigInt) -> bool {
    for (var i = 0u; i < NUM_WORDS; i ++) {
        if (coord.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Adds any two projective points
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-2002-bj
// 16M, 5 add, 4 sub
fn add_points(p1: Point, p2: Point) -> Point {
    var X1 = p1.x; var Y1 = p1.y; var Z1 = p1.z;
    var X2 = p2.x; var Y2 = p2.y; var Z2 = p2.z;

    if (is_zero(Z1)) {
        return p2;
    }
    if (is_zero(Z2)) {
        return p1;
    }

    var U1 = montgomery_product(&X1, &Z2);
    var U2 = montgomery_product(&X2, &Z1);
    var S1 = montgomery_product(&Y1, &Z2);
    var S2 = montgomery_product(&Y2, &Z1);
    var ZZ = montgomery_product(&Z1, &Z2);
    var T = fr_add(&U1, &U2);
    var M = fr_add(&S1, &S2);
    var U1U2 = montgomery_product(&U1, &U2);
    var TT = montgomery_product(&T, &T);
    var R = fr_sub(&TT, &U1U2);
    var F = montgomery_product(&ZZ, &M);
    var L = montgomery_product(&M, &F);
    var G = montgomery_product(&T, &L);
    var RR = montgomery_product(&R, &R);
    var W = fr_sub(&RR, &G);
    var FW = montgomery_product(&F, &W);
    var X3 = fr_add(&FW, &FW);
    var W2 = fr_add(&W, &W);
    var GW2 = fr_sub(&G, &W2);
    var RGW2 = montgomery_product(&R, &GW2);
    var LL = montgomery_product(&L, &L);
    var Y3 = fr_sub(&RGW2, &LL);

    var FF = montgomery_product(&F, &F);
    var FFF = montgomery_product(&FF, &F);
    var Z3 = fr_add(&FFF, &FFF);

    return Point(X3, Y3, Z3);
}

// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
fn double_point(p1: Point) -> Point {
    // 10M + 7add + 4 sub
    var x = p1.x; var y = p1.y; var z = p1.z;
    var XX = montgomery_product(&x, &x);
    var w = fr_add(&XX, &XX);
    w = fr_add(&w, &XX);
    var y1z1 = montgomery_product(&y, &z);
    var s = fr_add(&y1z1, &y1z1);
    var ss = montgomery_product(&s, &s);
    var sss = montgomery_product(&ss, &s);
    var R = montgomery_product(&y, &s);
    var RR = montgomery_product(&R, &R);
    var X1R = fr_add(&x, &R);
    var X1RX1R = montgomery_product(&X1R, &X1R);
    var xxrr = fr_add(&XX, &RR);
    var B = fr_sub(&X1RX1R, &xxrr);
    var ww = montgomery_product(&w, &w);
    var bb = fr_add(&B, &B);
    var h = fr_sub(&ww, &bb);
    var X3 = montgomery_product(&h, &s);
    var bh = fr_sub(&B, &h);
    var RRRR = fr_add(&RR, &RR);
    var wbh = montgomery_product(&w, &bh);
    var Y3 = fr_sub(&wbh, &RRRR);
    return Point(X3, Y3, sss);
}

