@group(0) @binding(0)
var<storage, read> scalars: array<u32>;
@group(0) @binding(1)
var<storage, read_write> result: array<u32>;

{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}
{{> montgomery_product_funcs }}

const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;

{{ > extract_word_from_bytes_le }}

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    var scalar_bytes: array<u32, 16>;
    for (var i = 0u; i < 16u; i ++) {
        scalar_bytes[15u - i] = scalars[id * 16 + i];
    }

    for (var i = 0u; i < NUM_WORDS - 1u; i ++) {
        result[id * NUM_WORDS + i] = extract_word_from_bytes_le(scalar_bytes, i);
    }

    result[id * NUM_WORDS + NUM_WORDS - 1u] = scalar_bytes[0] >> (((NUM_WORDS * WORD_SIZE - 256u) + 16u) - WORD_SIZE);
}
