@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@group(0) @binding(1)
var<storage, read> col_idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> row_ptr: array<u32>;

const NZ = {{ nz }}u;
const M = {{ m }}u;
const N = {{ n }}u;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    for (var i = 0u; i < NZ; i++) {
        let inner = col_idx[global_id.x + i] + 2u;
        output[inner] += 1;
    }
}