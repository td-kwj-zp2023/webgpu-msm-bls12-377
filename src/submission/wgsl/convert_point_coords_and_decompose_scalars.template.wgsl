{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}

// Bitwidth of each limb of the point coordinates
/*const WORD_SIZE = {{ word_size }}u;*/
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

// Number of chunks per scalar
const NUM_SUBTASKS = {{ num_subtasks }}u;

// Scalar chunk bitwidth
const CHUNK_SIZE = {{ chunk_size }}u;
const INPUT_SIZE = {{ input_size }};

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
    let id = gidx * {{ num_y_workgroups }} + gidy;

    // Convert x and y coordinates to byte arrays
    var x_bytes: array<u32, 16>;
    var y_bytes: array<u32, 16>;
    for (var i = 0u; i < 16u; i ++) {
        x_bytes[15u - i] = x_coords[id * 16 + i];
        y_bytes[15u - i] = y_coords[id * 16 + i];
    }

    // Convert the byte arrays to BigInts with word_size limbs
    var x_bigint: BigInt;
    var y_bigint: BigInt;
    for (var i = 0u; i < NUM_WORDS - 1u; i ++) {
        x_bigint.limbs[i] = extract_word_from_bytes_le(x_bytes, i, WORD_SIZE);
        y_bigint.limbs[i] = extract_word_from_bytes_le(y_bytes, i, WORD_SIZE);
    }

    let shift = (((NUM_WORDS * WORD_SIZE - 256u) + 16u) - WORD_SIZE);
    x_bigint.limbs[NUM_WORDS - 1u] = x_bytes[0] >> shift;
    y_bigint.limbs[NUM_WORDS - 1u] = y_bytes[0] >> shift;

    // Convert x and y coordinates to Montgomery form
    var r = get_r();
    point_x[id] = field_mul(&x_bigint, &r);
    point_y[id] = field_mul(&y_bigint, &r);

    // Decompose scalars
    var scalar_bytes: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        scalar_bytes[15u - i] = scalars[id * 16 + i];
    }

    var chunks_le: array<u32, {{ num_subtasks }}>;
    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        chunks_le[i] = extract_word_from_bytes_le(scalar_bytes, i, CHUNK_SIZE);
    }
    chunks_le[NUM_SUBTASKS - 1] = scalar_bytes[0] >> (((NUM_SUBTASKS * CHUNK_SIZE - 256u) + 16u) - CHUNK_SIZE);

    const l = {{ two_pow_chunk_size }}u;
    const index_shift = {{ index_shift }}u;

    var carry = 0u;
    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        var s = chunks_le[i] + carry;
        if (s >= index_shift) {
            s -= index_shift;
            carry = 1u;
        } else {
            carry = 0u;
        }
        chunks[id + i * INPUT_SIZE] = s + index_shift;
    }

/*
    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        let offset = i * INPUT_SIZE;
        chunks[id + offset] = extract_word_from_bytes_le(scalar_bytes, i, CHUNK_SIZE);
    }

    chunks[id + (NUM_SUBTASKS - 1) * INPUT_SIZE] = scalar_bytes[0] >> (((NUM_SUBTASKS * CHUNK_SIZE - 256u) + 16u) - CHUNK_SIZE);
    */
}
