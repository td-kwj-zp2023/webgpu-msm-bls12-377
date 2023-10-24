{{> bigint_struct }}
struct Point {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}

@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output[global_id.x] = points[global_id.x];
}
