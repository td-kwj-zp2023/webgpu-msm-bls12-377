{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}
{{> montgomery_product_funcs }}
{{ > extract_word_from_bytes_le_funcs }}

// Input buffers
@group(0) @binding(0)
var<storage, read> x_coords: array<u32>;
@group(0) @binding(1)
var<storage, read> y_coords: array<u32>;
@group(0) @binding(2)
var<storage, read> scalars: array<u32>;

// Output buffers
@group(0) @binding(3)
var<storage, read_write> point_x: array<BigInt>;
@group(0) @binding(4)
var<storage, read_write> point_y: array<BigInt>;
@group(0) @binding(5)
var<storage, read_write> chunks: array<u32>;

// Uniform buffer for parameters
@group(0) @binding(6)
var<uniform> input_size: u32;

const NUM_SUBTASKS = {{ num_subtasks }}u;

// Scalar chunk bitwidth
const CHUNK_SIZE = {{ chunk_size }}u;

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }}u + gidy;

    let INPUT_SIZE = input_size;
    let NUM_16_BIT_WORDS_PER_COORD = {{ num_16_bit_words_per_coord }}u;

    // Store the x and y coordinates as byte arrays for easier indexing
    var x_bytes: array<u32, {{ num_16_bit_words_per_coord }}>;
    var y_bytes: array<u32, {{ num_16_bit_words_per_coord }}>;
    for (var i = 0u; i < NUM_16_BIT_WORDS_PER_COORD; i ++) {
        x_bytes[NUM_16_BIT_WORDS_PER_COORD - 1 - i] = x_coords[id * NUM_16_BIT_WORDS_PER_COORD + i];
        y_bytes[NUM_16_BIT_WORDS_PER_COORD - 1 - i] = y_coords[id * NUM_16_BIT_WORDS_PER_COORD + i];
    }

    // Convert the byte arrays to BigInts with word_size limbs
    var x_bigint: BigInt;
    var y_bigint: BigInt;
    for (var i = 0u; i < NUM_WORDS - 1u; i ++) {
        x_bigint.limbs[i] = extract_word_from_coord_bytes_le(x_bytes, i, WORD_SIZE, NUM_16_BIT_WORDS_PER_COORD);
        y_bigint.limbs[i] = extract_word_from_coord_bytes_le(y_bytes, i, WORD_SIZE, NUM_16_BIT_WORDS_PER_COORD);
    }

    let shift = (((NUM_WORDS * WORD_SIZE - 256u) + 16u) - WORD_SIZE);
    x_bigint.limbs[NUM_WORDS - 1u] = x_bytes[0] >> shift;
    y_bigint.limbs[NUM_WORDS - 1u] = y_bytes[0] >> shift;

    // Convert x and y coordinates to Montgomery form
    var r = get_r();
    point_x[id] = field_mul(&x_bigint, &r);
    point_y[id] = field_mul(&y_bigint, &r);

    // Note that we only compute the t and z coordinates in the SMVP shader
    // as WebGPU limits the number of buffers per shader to 8.

    // Decompose scalars
    var scalar_bytes: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        let s = scalars[id * 8 + i];
        let hi = s >> 16u;
        let lo = s & 65535u;
        scalar_bytes[15 - (i * 2)] = lo;
        scalar_bytes[15 - (i * 2) - 1] = hi;
    }

    // Extract scalar chunks and store them in chunks_arr
    var chunks_arr: array<u32, {{ num_subtasks }}>;
    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        let offset = i * INPUT_SIZE;
        chunks_arr[i] = extract_word_from_bytes_le(scalar_bytes, i, CHUNK_SIZE);
    }
    chunks_arr[NUM_SUBTASKS - 1] = scalar_bytes[0] >> (((NUM_SUBTASKS * CHUNK_SIZE - 256u) + 16u) - CHUNK_SIZE);

    // Iterate through chunks_arr to compute the signed indices
    let l = {{ num_columns }}u;
    let s = l / 2u;

    var signed_slices: array<i32, {{ num_subtasks }}>;
    var carry = 0u;
    for (var i = 0u; i < NUM_SUBTASKS; i ++) {
        signed_slices[i] = i32(chunks_arr[i] + carry);
        if (signed_slices[i] >= i32(s)) {
            signed_slices[i] = (i32(l) - signed_slices[i]) * -1i;
            carry = 1u;
        } else {
            carry = 0u;
        }
    }

    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        let offset = i * INPUT_SIZE;

        // Note that we add s (half_num_columns) to the bucket index so we
        // don't store negative values, while retaining information about the
        // sign of the original index.
        chunks[id + offset] = u32(signed_slices[i]) + s;
    }

    {{{ recompile }}}
}
