
{{> montgomery_product_funcs }}

// /*
//   var a = field_multiply(p1.x, p2.x);
//   var b = field_multiply(p1.y, p2.y);
//   var c = field_multiply(EDWARDS_D, field_multiply(p1.t, p2.t));
//   var d = field_multiply(p1.z, p2.z);
//   var p1_added = field_add(p1.x, p1.y);
//   var p2_added = field_add(p2.x, p2.y);
//   var e = field_multiply(field_add(p1.x, p1.y), field_add(p2.x, p2.y));
//   e = field_sub(e, a);
//   e = field_sub(e, b);
//   var f = field_sub(d, c);
//   var g = field_add(d, c);
//   var a_neg = mul_by_a(a);
//   var h = field_sub(b, a_neg);
//   var added_x = field_multiply(e, f);
//   var added_y = field_multiply(g, h);
//   var added_t = field_multiply(e, h);
//   var added_z = field_multiply(f, g);
//   return Point(added_x, added_y, added_t, added_z);
//   */

fn get_edwards_d() -> BigInt {
    var d: BigInt;
    d.limbs[0] = 1953u;
    d.limbs[1] = 8111u;
    d.limbs[2] = 8191u;
    d.limbs[3] = 767u;
    d.limbs[4] = 2552u;
    d.limbs[5] = 254u;
    d.limbs[6] = 8150u;
    d.limbs[7] = 7705u;
    d.limbs[8] = 467u;
    d.limbs[9] = 6341u;
    d.limbs[10] = 923u;
    d.limbs[11] = 7568u;
    d.limbs[12] = 3738u;
    d.limbs[13] = 5751u;
    d.limbs[14] = 6346u;
    d.limbs[15] = 6298u;
    d.limbs[16] = 1166u;
    d.limbs[17] = 3473u;
    d.limbs[18] = 3231u;
    d.limbs[19] = 3606u;
    return d;
}

fn add_points(p1: Point, p2: Point) -> Point {
    
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

    var p1_added = fr_add(&p1x, &p1y);
    var p2_added = fr_add(&p2x, &p2y);
    var e = montgomery_product(&p1_added, &p2_added);

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