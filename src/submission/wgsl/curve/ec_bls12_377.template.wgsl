struct Point {
  x: BigInt,
  y: BigInt,
  z: BigInt
}

// Assumes that p1.z and p2.z are both equal to 1
// 10M + 15add
fn add_points_zs_1(p1: Point, p2: Point) -> Point {
    var X1 = p1.x; var Y1 = p1.y;
    var X2 = p2.x; var Y2 = p2.y;

    var T = fr_add(&X1, &X2);
    var TT = montgomery_product(&T, &T);
    var F = fr_add(&Y1, &Y2);

    var U1U2 = montgomery_product(&X1, &X2);
    var R = fr_sub(&TT, &U1U2);

    var L = montgomery_product(&F, &F);
    var LL = montgomery_product(&L, &L);

    var TL = fr_add(&T, &L);
    var TLTL = montgomery_product(&TL, &TL);
    var TTLL = fr_add(&TT, &LL);
    var G = fr_sub(&TLTL, &TTLL);

    var RR = montgomery_product(&R, &R);
    var RR2 = fr_add(&RR, &RR);
    var W = fr_sub(&RR2, &G);

    var FW = montgomery_product(&F, &W);
    var X3 = fr_add(&FW, &FW);

    var LL2 = fr_add(&LL, &LL);
    var W2 = fr_add(&W, &W);
    var GW2 = fr_sub(&G, &W2);
    var RGW2 = montgomery_product(&R, &GW2);
    var Y3 = fr_sub(&RGW2, &LL2);

    var F2 = fr_add(&F, &F);
    var F4 = fr_add(&F2, &F2);
    var FS = montgomery_product(&F, &F);
    var Z3 = montgomery_product(&F4, &FS);

    return Point(X3, Y3, Z3);
}

fn is_zero(coord: BigInt) -> bool {
    for (var i = 0u; i < NUM_WORDS; i ++) {
        if (coord.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Adds any two projective points
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-2007-bl
// 16M + 15add
fn add_points(p1: Point, p2: Point) -> Point {
    var X1 = p1.x; var Y1 = p1.y; var Z1 = p1.z;
    var X2 = p2.x; var Y2 = p2.y; var Z2 = p2.z;

    // TODO: optimise this within SMVP
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
    var TT = montgomery_product(&T, &T);
    var M = fr_add(&S1, &S2);

    var U1U2 = montgomery_product(&U1, &U2);
    var R = fr_sub(&TT, &U1U2);

    var F = montgomery_product(&ZZ, &M);
    var L = montgomery_product(&M, &F);
    var LL = montgomery_product(&L, &L);

    var TL = fr_add(&T, &L);
    var TLTL = montgomery_product(&TL, &TL);
    var TTLL = fr_add(&TT, &LL);
    var G = fr_sub(&TLTL, &TTLL);

    var RR = montgomery_product(&R, &R);
    var RR2 = fr_add(&RR, &RR);
    var W = fr_sub(&RR2, &G);

    var FW = montgomery_product(&F, &W);
    var X3 = fr_add(&FW, &FW);

    var LL2 = fr_add(&LL, &LL);
    var W2 = fr_add(&W, &W);
    var GW2 = fr_sub(&G, &W2);
    var RGW2 = montgomery_product(&R, &GW2);
    var Y3 = fr_sub(&RGW2, &LL2);

    var F2 = fr_add(&F, &F);
    var F4 = fr_add(&F2, &F2);
    var FS = montgomery_product(&F, &F);
    var Z3 = montgomery_product(&F4, &FS);

    return Point(X3, Y3, Z3);
}

// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
fn double_point(p1: Point) -> Point {
    // 10M + 11add
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

