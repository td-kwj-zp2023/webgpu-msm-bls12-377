
{{> montgomery_product_funcs }}

fn add_points(p1: Point, p2: Point) -> Point {
    
    var p1x = p1.x;
    var p2x = p2.x;
    var a = montgomery_product(&p1x, &p2x);
    
    var p1y = p1.y;
    var p2y = p2.y;
    var b = montgomery_product(&p1y, &p2y);

    var p1t = p1.t;
    var p2t = p1.t;
    var t2 = montgomery_product(&p1t, &p2t);

    var EDWARDS_D: BigInt;
    EDWARDS_D.limbs[0] = 3021u;
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