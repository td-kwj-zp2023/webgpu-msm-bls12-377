{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> montgomery_product_funcs }}

@group(0) @binding(0)
var<storage, read> x_coords: array<BigInt>;
@group(0) @binding(1)
var<storage, read> y_coords: array<BigInt>;
@group(0) @binding(2)
var<storage, read_write> out_x_coords: array<BigInt>;
@group(0) @binding(3)
var<storage, read_write> out_y_coords: array<BigInt>;

struct SWPoint {
  x: BigInt,
  y: BigInt,
  z: BigInt
}

// Assumes that p1.z and p2.z are both equal to 1
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-mmadd-1998-cmo
fn add_affine_points(p1: SWPoint, p2: SWPoint) -> SWPoint {
    /*
      u = Y2-Y1
      uu = u2
      v = X2-X1
      vv = v2
      vvv = v*vv
      R = vv*X1
      A = uu-vvv-2*R
      X3 = v*A
      Y3 = u*(R-A)-vvv*Y1
      Z3 = vvv
    */
}

// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-1998-cmo-2
fn add_points(p1: SWPoint, p2: SWPoint) -> SWPoint {
    /*
      Y1Z2 = Y1*Z2
      X1Z2 = X1*Z2
      Z1Z2 = Z1*Z2
      u = Y2*Z1-Y1Z2
      uu = u2
      v = X2*Z1-X1Z2
      vv = v2
      vvv = v*vv
      R = vv*X1Z2
      A = uu*Z1Z2-vvv-2*R
      X3 = v*A
      Y3 = u*(R-A)-vvv*Y1Z2
      Z3 = vvv*Z1Z2
    */
}

// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-bl
fn double_point(p1: SWPoint) -> SWPoint {
    /*
      XX = X12
      ZZ = Z12
      w = a*ZZ+3*XX
      s = 2*Y1*Z1
      ss = s2
      sss = s*ss
      R = Y1*s
      RR = R2
      B = (X1+R)2-XX-RR
      h = w2-2*B
      X3 = h*s
      Y3 = w*(B-h)-2*RR
      Z3 = sss
    */
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var id = global_id.x;

    var x = x_coords[id];
    var y = y_coords[id];

    var z: BigInt;
    z.limbs[0] = 1u;

    var pt = SWPoint(x, y , z);

    out_x_coords[id] = x;
    out_y_coords[id] = y;
}
