
{{> structs }}
{{> montgomery_product_funcs }}
{{> field_functions }}
{{> bigint_functions }}
{{> curve_parameters }}
{{> curve_functions }}

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

// fn add_points(p1: Point, p2: Point) -> Point {
//     var p1x = p1.x;
//     var p2x = p2.x;
//     var a = montgomery_product(&p1x, &p2x);

//     var p1y = p1.y;
//     var p2y = p2.y;
//     var b = montgomery_product(&p1y, &p2y);

//     var p1t = p1.t;
//     var p2t = p2.t;
//     var t2 = montgomery_product(&p1t, &p2t);

//     var EDWARDS_D = get_edwards_d();
//     var c = montgomery_product(&EDWARDS_D, &t2);

//     var p1z = p1.z;
//     var p2z = p2.z;
//     var d = montgomery_product(&p1z, &p2z);
//     var xpy = fr_add(&p1x, &p1y);
//     var xpy2 = fr_add(&p2x, &p2y);
//     var e = montgomery_product(&xpy, &xpy2);
//     e = fr_sub(&e, &a);
//     e = fr_sub(&e, &b);

//     var f = fr_sub(&d, &c);
//     var g = fr_add(&d, &c);

//     var p = get_p();
//     var a_neg = fr_sub(&p, &a);

//     var h = fr_sub(&b, &a_neg);
//     var added_x = montgomery_product(&e, &f);
//     var added_y = montgomery_product(&g, &h);
//     var added_t = montgomery_product(&e, &h);
//     var added_z = montgomery_product(&f, &g);

//     return Point(added_x, added_y, added_t, added_z);
// }

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // During the first pass, store the point at infinity in the output buffer.
    if (loop_index == 0u) {
        for (var i = 0u; i < arrayLength(&output); i ++) {
            var y: BigInt = get_r();
            var z: BigInt = get_r();

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
