@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var a: array<u32, 32>;
    var b: array<u32, 32>;
    var c: array<u32, 32>;
    var d: array<u32, 32>;

    for (var i = 0u; i < 32u; i ++) {
        a[i] = i;
        b[i] = i;
        c[i] = a[i] * b[i];
        d[i] = c[i] * b[i];
    }

    data[global_id.x] = a[0] + b[0] + data[global_id.x];
}

