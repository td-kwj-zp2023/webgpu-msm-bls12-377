@group(0) @binding(0)
var<storage, read> scalars: array<u32>;
@group(0) @binding(1)
var<storage, read_write> result: array<u32>;

const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;

fn extract_word_from_bytes_le(
    input: array<u32, 16>,
    word_idx: u32
) -> u32 {
    var word = 0u;
    let word_size = WORD_SIZE;
    let start_byte_idx = 15u - ((word_idx * word_size + word_size) / 16u);
    let end_byte_idx = 15u - ((word_idx * word_size) / 16u);

    let start_byte_offset = (word_idx * word_size + word_size) % 16u;
    let end_byte_offset = (word_idx * word_size) % 16u;

    var mask = 0u;
    if (start_byte_offset > 0u) {
        mask = (2u << (start_byte_offset - 1u)) - 1u;
    }
    if (start_byte_idx == end_byte_idx) {
        word = (input[start_byte_idx] & mask) >> end_byte_offset;
    } else {
        word = (input[start_byte_idx] & mask) << (16u - end_byte_offset);
        word += input[end_byte_idx] >> end_byte_offset;
    }

    return word;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var scalar_bytes: array<u32, 16>;
    for (var i = 0u; i < 16; i ++) {
        scalar_bytes[15 - i] = scalars[global_id.x * 16 + i];
    }

    for (var i = 0u; i < NUM_WORDS - 1u; i ++) {
        result[global_id.x * NUM_WORDS + i] = extract_word_from_bytes_le(scalar_bytes, i);
    }
    /*result[global_id.x * NUM_WORDS + NUM_WORDS - 1u] = scalar_bytes[0] >> (NUM_WORDS - WORD_SIZE);*/
    result[global_id.x * NUM_WORDS + NUM_WORDS - 1u] = scalar_bytes[0] >> (((NUM_WORDS * WORD_SIZE - 256) + 16) - WORD_SIZE);
}
