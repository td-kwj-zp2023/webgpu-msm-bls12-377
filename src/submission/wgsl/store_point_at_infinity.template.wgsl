@group(0) @binding(0)
var<storage, read_write> output: array<ETEPoint>;

struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

struct ETEPoint {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Store the point at infinity in each point in the output buffer
    var x: BigInt;
    var y: BigInt;
    var t: BigInt;
    var z: BigInt;

    y.limbs[0] = 1u;
    z.limbs[0] = 1u;

    var inf: ETEPoint;
    inf.x = x;
    inf.y = y;
    inf.t = t;
    inf.z = z;
    output[global_id.x] = inf;
}
