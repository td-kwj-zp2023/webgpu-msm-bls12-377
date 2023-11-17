@group(0) @binding(0)
var<storage, read> scalars: array<u32>;
@group(0) @binding(1)
var<storage, read_write> result: array<u32>;

const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;

fn extract_word_from_bytes_le(
    bytes_start_idx: u32,
    bytes_end_idx: u32,
    word_idx: u32
) -> u32 {
    let input_len = 32u;
    let start_byte_idx = input_len - 1u - ((word_idx * WORD_SIZE + WORD_SIZE) / 8);
    let end_byte_idx = input_len - 1u - ((word_idx * WORD_SIZE) / 8u);
    let start_byte_offset = (word_idx * WORD_SIZE + WORD_SIZE) % 8u;
    let end_byte_offset = (word_idx * WORD_SIZE) % 8u;

    var sum = 0u;
    for (var i = start_byte_idx; i < end_byte_idx + 1u; i ++) {
        let input = scalars[bytes_start_idx + i];
        if (i == start_byte_idx) {
            let mask = (2u << (start_byte_offset - 1u)) - 1u;
            sum += input & mask;
        } else if (i == end_byte_idx) {
            sum = sum << (8u - end_byte_offset);
            sum += input >> end_byte_offset;
        } else {
            sum = sum << 8u;
            sum += input;
        }
    }
    return sum;
}

@compute
@workgroup_size(2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bytes_start_idx = global_id.x * 32;
    let bytes_end_idx = global_id.x * 32 + 32;

    for (var i = 1u; i < NUM_WORDS; i ++) {
        result[global_id.x + i] = extract_word_from_bytes_le(bytes_start_idx, bytes_end_idx, i);
    }
}
