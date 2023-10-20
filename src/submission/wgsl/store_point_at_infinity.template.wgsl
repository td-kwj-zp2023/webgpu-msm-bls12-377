@group(0) @binding(0)
var<storage, read_write> output: array<Point>;

struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

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

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Store the point at infinity in each point in the output buffer
    var x: BigInt;
    var y: BigInt = get_r();
    var t: BigInt;
    var z: BigInt = get_r();

    var inf: Point;
    inf.x = x;
    inf.y = y;
    inf.t = t;
    inf.z = z;
    output[global_id.x] = inf;
}
