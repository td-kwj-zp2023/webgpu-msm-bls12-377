import mustache from 'mustache'
import { BigIntPoint } from "../../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../../reference/utils/FieldMath";
import {
    get_device,
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
    execute_pipeline,
} from '../gpu'
import {
    to_words_le,
    gen_p_limbs,
    gen_r_limbs,
    gen_mu_limbs,
    u8s_to_bigints,
    u8s_to_numbers,
    u8s_to_numbers_32,
    numbers_to_u8s_for_gpu,
    bigints_to_16_bit_words_for_gpu,
    bigints_to_u8_for_gpu,
    compute_misc_params,
    decompose_scalars,
} from '../utils'
import assert from 'assert'

import convert_point_coords_shader from '../wgsl/convert_point_coords.template.wgsl'
import extract_word_from_bytes_le_funcs from '../wgsl/extract_word_from_bytes_le.template.wgsl'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_funcs from '../wgsl/bigint/bigint.template.wgsl'
import field_funcs from '../wgsl/field/field.template.wgsl'
import ec_funcs from '../wgsl/curve/ec.template.wgsl'
import barrett_funcs from '../wgsl/barrett.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery/mont_pro_product.template.wgsl'
import decompose_scalars_shader from '../wgsl/decompose_scalars.template.wgsl'
import gen_csr_precompute_shader from '../wgsl/gen_csr_precompute.template.wgsl'
import preaggregation_stage_1_shader from '../wgsl/preaggregation_stage_1.template.wgsl'
import preaggregation_stage_2_shader from '../wgsl/preaggregation_stage_2.template.wgsl'
import compute_row_ptr_shader from '../wgsl/compute_row_ptr_shader.template.wgsl'

const fieldMath = new FieldMath()

// Hardcode params for word_size = 13
const p = BigInt('8444461749428370424248824938781546531375899335154063827935233455917409239041')
const word_size = 13
const params = compute_misc_params(p, word_size)
const n0 = params.n0
const num_words = params.num_words
const r = params.r
const rinv = params.rinv

/*
 * End-to-end implementation of the cuZK MSM algorithm.
 */
export const cuzk_gpu = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    // Determine the optimal window size dynamically based on a static analysis 
    // of varying input sizes. This will be determined using a seperate function.   
    const input_size = scalars.length
    const chunk_size = 16
    const num_subtasks = Math.ceil(256 / chunk_size)
    const num_rows_per_subtask = 16

    // Each pass must use the same GPUDevice and GPUCommandEncoder, or else
    // storage buffers can't be reused across compute passes
    const device = await get_device()
    const commandEncoder = device.createCommandEncoder()

    // Convert the affine points to Montgomery form in the GPU
    const { point_x_y_sb, point_t_z_sb } =
        await convert_point_coords_to_mont_gpu(
            device,
            commandEncoder,
            baseAffinePoints,
            num_words, 
            word_size,
            false,
        )

    // Decompose the scalars
    const scalar_chunks_sb = await decompose_scalars_gpu(
        device,
        commandEncoder,
        scalars,
        num_subtasks,
        chunk_size,
        false,
    )

    for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
        // use debug_idx to debug any particular subtask_idx
        //const debug_idx = subtask_idx === 0

        // TODO: if debug is set to true in any invocations within a loop, the
        // sanity check will fail on the second iteration, because the
        // commandEncoder's finish() function has been used. To correctly
        // sanity-check these outputs, do so in a separate test file.
        const {
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
        } = await csr_precompute_gpu(
            device,
            commandEncoder,
            input_size,
            num_subtasks,
            subtask_idx,
            chunk_size,
            scalar_chunks_sb,
            false,
        )

        const {
            new_point_x_y_sb,
            new_point_t_z_sb,
        } = await pre_aggregation_stage_1_gpu(
            device,
            commandEncoder,
            input_size,
            point_x_y_sb,
            point_t_z_sb,
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
            false,
        )

        const new_scalar_chunks_sb = await pre_aggregation_stage_2_gpu(
            device,
            commandEncoder,
            input_size,
            scalar_chunks_sb,
            cluster_start_indices_sb,
            new_point_indices_sb,
            false,
        )

        const row_ptr_sb = await compute_row_ptr(
            device,
            commandEncoder,
            input_size,
            num_subtasks,
            num_rows_per_subtask,
            new_point_indices_sb,
            false,
            //debug_idx
        )
        //if (debug_idx) { break }
        // TODO: perform transposition
        // TODO: perform SMVP
        // TODO: perform bucket aggregation
    }
    // TODO: perform Horner's rule

    device.destroy()

    return { x: BigInt(1), y: BigInt(0) }
}

/*
 * Convert the affine points to Montgomery form

 * ASSUMPTION: the vast majority of WebGPU-enabled consumer devices have a
 * maximum buffer size of at least 268435456 bytes.
 * 
 * The default maximum buffer size is 268435456 bytes. Since each point
 * consumes 320 bytes, a maximum of around 2 ** 19 points can be stored in a
 * single buffer. If, however, we use 4 buffers - one for each point coordiante
 * X, Y, T, and Z - we can support up an input size of up to 2 ** 21 points.
 * Our implementation, however, will only support up to 2 ** 20 points as that
 * is the maximum input size for the ZPrize competition.
 * 
 * The test harness readme at https://github.com/demox-labs/webgpu-msm states:
 * "The submission should produce correct outputs on input vectors with length
 * up to 2^20. The evaluation will be using input randomly sampled from size
 * 2^16 ~ 2^20."
*/
export const convert_point_coords_to_mont_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    baseAffinePoints: BigIntPoint[],
    num_words: number,
    word_size: number,
    debug = false,
): Promise<{
    point_x_y_sb: GPUBuffer,
    point_t_z_sb: GPUBuffer,
}> => {
    const input_size = baseAffinePoints.length

    // An affine point only contains X and Y points.
    const x_y_coords = Array(input_size * 2).fill(BigInt(0))
    for (let i = 0; i < input_size; i ++) {
        x_y_coords[i * 2] = baseAffinePoints[i].x
        x_y_coords[i * 2 + 1] = baseAffinePoints[i].y
    }

    // Convert points to bytes (performs ~2x faster than
    // `bigints_to_16_bit_words_for_gpu`)
    const x_y_coords_bytes = bigints_to_u8_for_gpu(x_y_coords, 16, 16)

    // Input buffers
    const x_y_coords_sb = create_and_write_sb(device, x_y_coords_bytes)

    // Output buffers
    const point_x_y_sb = create_sb(device, input_size * 2 * num_words * 4)
    const point_t_z_sb = create_sb(device, input_size * 2 * num_words * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'storage',
            'storage',
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            x_y_coords_sb,
            point_x_y_sb,
            point_t_z_sb,
        ],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = baseAffinePoints.length / num_x_workgroups / workgroup_size

    const shaderCode = genConvertPointCoordsShaderCode(
        workgroup_size,
        num_y_workgroups,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [
                point_x_y_sb,
                point_t_z_sb,
            ],
        )
        
        const computed_x_y_coords = u8s_to_bigints(data[0], num_words, word_size)
        const computed_t_z_coords = u8s_to_bigints(data[1], num_words, word_size)

        for (let i = 0; i < input_size; i ++) {
            const expected_x = baseAffinePoints[i].x * r % p
            const expected_y = baseAffinePoints[i].y * r % p
            const expected_t = (baseAffinePoints[i].x * baseAffinePoints[i].y * r) % p
            const expected_z = r % p

            if (!(
                expected_x === computed_x_y_coords[i * 2] 
                && expected_y === computed_x_y_coords[i * 2 + 1] 
                && expected_t === computed_t_z_coords[i * 2] 
                && expected_z === computed_t_z_coords[i * 2 + 1]
            )) {
                console.log('mismatch at', i)
                debugger
                break
            }
        }
    }

    // Destroy unused buffers
    x_y_coords_sb.destroy()

    return { point_x_y_sb, point_t_z_sb }
}

const genConvertPointCoordsShaderCode = (
    workgroup_size: number,
    num_y_workgroups: number,
) => {
    const mask = BigInt(2) ** BigInt(word_size) - BigInt(1)
    const two_pow_word_size = 2 ** word_size
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const mu_limbs = gen_mu_limbs(p, num_words, word_size)
    const p_bitlength = p.toString(2).length
    const slack = num_words * word_size - p_bitlength
        const shaderCode = mustache.render(
        convert_point_coords_shader,
        {
            workgroup_size,
            num_y_workgroups,
            num_words,
            word_size,
            n0,
            mask,
            two_pow_word_size,
            p_limbs,
            r_limbs,
            mu_limbs,
            w_mask: (1 << word_size) - 1,
            slack,
            num_words_mul_two: num_words * 2,
            num_words_plus_one: num_words + 1,
        },
        {
            structs,
            bigint_funcs,
            field_funcs,
            barrett_funcs,
            montgomery_product_funcs,
            extract_word_from_bytes_le_funcs,
        },
    )
    return shaderCode
}

export const decompose_scalars_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    scalars: bigint[],
    num_subtasks: number,
    chunk_size: number,
    debug = false,
): Promise<GPUBuffer> => {
    const input_size = scalars.length
    assert(num_subtasks * chunk_size === 256)

    // Convert scalars to bytes
    const scalars_bytes = bigints_to_16_bit_words_for_gpu(scalars)

    // Input buffers
    const scalars_sb = create_and_write_sb(device, scalars_bytes)

    // Output buffer(s)
    const chunks_sb = create_sb(device, input_size * num_subtasks * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        ['read-only-storage', 'storage'],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [scalars_sb, chunks_sb],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = input_size / workgroup_size / num_x_workgroups

    const shaderCode = genDecomposeScalarsShaderCode(
        workgroup_size,
        num_y_workgroups,
        num_subtasks,
        chunk_size, 
        input_size
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1)

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [chunks_sb],
        )

        const computed_chunks = u8s_to_numbers(data[0])

        const all_chunks: Uint16Array[] = []

        const expected: number[] = Array(scalars.length * num_subtasks).fill(0)
        for (let i = 0; i < scalars.length; i ++) {
            const chunks = to_words_le(scalars[i], num_subtasks, chunk_size)
            all_chunks.push(chunks)
        }
        for (let i = 0; i < chunk_size; i ++) {
            for (let j = 0; j < scalars.length; j ++) {
                expected[j * chunk_size + i] = all_chunks[j][i]
            }
        }

        const decompose_scalars_original = decompose_scalars(scalars, num_subtasks, chunk_size)

        if (computed_chunks.length !== expected.length) {
            throw Error('output size mismatch')
        }

        for (let j = 0; j < decompose_scalars_original.length; j++) {
            let z = 0;
            for (let i = j * input_size; i < (j + 1) * input_size; i++) {
                if (computed_chunks[i] !== decompose_scalars_original[j][z]) {
                    throw Error(`scalar decomp mismatch at ${i}`)
                }
                z++;
            }
        }
    }

    // Destroy unused buffers
    scalars_sb.destroy()

    return chunks_sb
}

const genDecomposeScalarsShaderCode = (
    workgroup_size: number,
    num_y_workgroups: number,
    num_subtasks: number,
    chunk_size: number,
    input_size: number
) => {
    const shaderCode = mustache.render(
        decompose_scalars_shader,
        {
            workgroup_size,
            num_y_workgroups,
            num_subtasks,
            chunk_size,
            input_size,
        },
        {
            extract_word_from_bytes_le_funcs,
        },
    )
    return shaderCode
}

export const csr_precompute_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    input_size: number,
    num_subtasks: number,
    subtask_idx: number,
    chunk_size: number,
    scalar_chunks_sb: GPUBuffer,
    debug = true,
): Promise<{
    new_point_indices_sb: GPUBuffer,
    cluster_start_indices_sb: GPUBuffer,
    cluster_end_indices_sb: GPUBuffer,
}> => {
    /*
    // Test values
    const test_scalar_chunks = 
        [
            1, 1, 1, 1, 0, 1, 6, 7, 1, 1, 1, 1, 4, 4, 6, 7,
            1, 1, 1, 1, 0, 1, 6, 7, 1, 1, 1, 1, 4, 4, 6, 7,
        ]
    const test_scalar_chunks_bytes = numbers_to_u8s_for_gpu(test_scalar_chunks)
    scalar_chunks_sb = create_and_write_sb(device, test_scalar_chunks_bytes)
    input_size = test_scalar_chunks.length
    num_subtasks = 2
    subtask_idx = 1

    const test_scalar_chunks_bytes = numbers_to_u8s_for_gpu(TEST_CHUNKS)
    scalar_chunks_sb = create_and_write_sb(device, test_scalar_chunks_bytes)
    subtask_idx = 1
    */

    // This is a serial operation, so only 1 shader should be used
    const num_x_workgroups = 1
    const num_y_workgroups = 1 

    const num_chunks = input_size / num_subtasks

    // Adjust max_cluster_size based on the input size
    let max_cluster_size = 4
    if (input_size >= 2 ** 20) {
        max_cluster_size = 2
    } else if (input_size >= 2 ** 16) {
        max_cluster_size = 3
    }
    const max_chunk_val = 2 ** chunk_size
    const overflow_size = num_chunks - max_cluster_size

    // Output buffers
    const new_point_indices_sb = create_sb(device, num_chunks * 4)
    const cluster_start_indices_sb = create_sb(device, num_chunks * 4)
    const cluster_end_indices_sb = create_sb(device, num_chunks * 4)
    const map_sb = create_sb(device, (max_cluster_size + 1) * max_chunk_val * 4)
    const overflow_sb = create_sb(device, overflow_size * 4)
    const keys_sb = create_sb(device, max_chunk_val * 4)
    const subtask_idx_sb = create_and_write_sb(device, numbers_to_u8s_for_gpu([subtask_idx]))

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage', 'read-only-storage', 'storage', 'storage',
            'storage', 'storage', 'storage', 'storage',
        ]
    )

    // Reuse the output buffer from the scalar decomp step as one of the input buffers
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            scalar_chunks_sb,
            subtask_idx_sb,
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
            map_sb,
            overflow_sb,
            keys_sb
        ],
    )

    const shaderCode = genCsrPrecomputeShaderCode(
        num_y_workgroups,
        max_chunk_val,
		input_size,
        num_subtasks,
        max_cluster_size,
        overflow_size,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [
                new_point_indices_sb,
                cluster_start_indices_sb,
                cluster_end_indices_sb,
                scalar_chunks_sb,
                map_sb,
            ],
        )

        const [
            new_point_indices,
            cluster_start_indices,
            cluster_end_indices,
            scalar_chunks,
        ] = data.map(u8s_to_numbers_32)

        verify_gpu_precompute_output(
            input_size,
            subtask_idx,
            num_subtasks,
            max_cluster_size,
            overflow_size,
            scalar_chunks,
            new_point_indices,
            cluster_start_indices,
            cluster_end_indices,
        )
    }

    // Destroy unused buffers
    subtask_idx_sb.destroy()
    map_sb.destroy()
    overflow_sb.destroy()
    keys_sb.destroy()

    return { new_point_indices_sb, cluster_start_indices_sb, cluster_end_indices_sb }
}

const verify_gpu_precompute_output = (
    input_size: number,
    subtask_idx: number,
    num_subtasks: number,
    max_cluster_size: number,
    overflow_size: number,
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
    cluster_end_indices: number[],
) => {
    const num_chunks = input_size / num_subtasks

    // During testing
    if (scalar_chunks.length < input_size) {
        const pad = Array(subtask_idx * num_chunks).fill(0)
        scalar_chunks = pad.concat(scalar_chunks)
    }

    const scalar_chunks_for_this_subtask = scalar_chunks.slice(
        subtask_idx * num_chunks,
        subtask_idx * num_chunks + num_chunks,
    )

    // Check that the values in new_point_indices can be used to reconstruct a
    // list of scalar chunks which, when sorted, match the sorted scalar chunks
    const reconstructed = Array(num_chunks).fill(0)

    for (let i = 0; i < num_chunks; i ++) {
        if (i > 0 && new_point_indices[i] === 0) {
            break
        }
        reconstructed[i] = scalar_chunks[new_point_indices[i]]
    }

    const sc_copy = scalar_chunks_for_this_subtask.map((x) => Number(x))
    const r_copy = reconstructed.map((x) => Number(x))
    sc_copy.sort((a, b) => a - b)
    r_copy.sort((a, b) => a - b)

    //console.log('chunks:', scalar_chunks_for_this_subtask.toString())
    //console.log('reconstructed:', reconstructed.toString())
    //console.log('sc_copy:', sc_copy.toString())
    //console.log('r_copy:', r_copy.toString())
    //if (sc_copy.toString() !== r_copy.toString()) {
        //debugger
        //assert(false)
    //}
    assert(sc_copy.toString() === r_copy.toString(), 'new_point_indices invalid')

    // Ensure that cluster_start_indices and cluster_end_indices have
    // the correct structure
    assert(cluster_start_indices.length === cluster_end_indices.length)
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        // start <= end
        assert(cluster_start_indices[i] <= cluster_end_indices[i], `invalid cluster index at ${i}`)
    }
 
    // Check that the cluster start- and end- indices respect
    // max_cluster_size 
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        // end - start <= max_cluster_size
        const d = cluster_end_indices[i] - cluster_start_indices[i]
        assert(d <= max_cluster_size)
    }
 
    // Generate random "points" and compute their linear combination
    // without any preaggregation, then compare the result using an algorithm
    // that uses preaggregation first
    const random_points: bigint[] = []
    for (let i = 0; i < input_size; i ++) {
        //const r = BigInt(Math.floor(Math.random() * 100000000))
        const r = BigInt(1)
        random_points.push(r)
    }

    // Calcualte the linear combination naively
    let lc_result = BigInt(0)
    for (let i = 0; i < num_chunks; i ++) {
        const prod = BigInt(scalar_chunks_for_this_subtask[i]) * random_points[i]
        lc_result += BigInt(prod)
    }

    // Calculate the linear combination with preaggregation
    let preagg_result = BigInt(0)
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        const end_idx = cluster_end_indices[i]

        if (end_idx === 0) {
            break
        }

        let point = BigInt(random_points[new_point_indices[start_idx]])

        for (let idx = cluster_start_indices[i] + 1; idx < end_idx; idx ++) {
            point += BigInt(random_points[new_point_indices[idx]])
        }

        preagg_result += point * BigInt(scalar_chunks[new_point_indices[start_idx]])
    }

    assert(preagg_result === lc_result, 'result mismatch')
}

const genCsrPrecomputeShaderCode = (
    num_y_workgroups: number,
    max_chunk_val: number,
	input_size: number,
    num_subtasks: number,
    max_cluster_size: number,
    overflow_size: number,
) => {
    const shaderCode = mustache.render(
        gen_csr_precompute_shader,
        {
            num_y_workgroups,
            num_subtasks,
            max_cluster_size,
            max_cluster_size_plus_one: max_cluster_size + 1,
            max_chunk_val,
            num_chunks: input_size / num_subtasks,
            overflow_size,
        },
        {},
    )
    return shaderCode
}

export const pre_aggregation_stage_1_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    input_size: number,
    point_x_y_sb: GPUBuffer,
    point_t_z_sb: GPUBuffer,
    new_point_indices_sb: GPUBuffer,
    cluster_start_indices_sb: GPUBuffer,
    cluster_end_indices_sb: GPUBuffer,
    debug = false,
): Promise<{
    new_point_x_y_sb: GPUBuffer,
    new_point_t_z_sb: GPUBuffer,
}> => {
    const new_point_x_y_sb = create_sb(device, input_size * 2 * num_words * 4)
    const new_point_t_z_sb = create_sb(device, input_size * 2 * num_words * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            point_x_y_sb,
            point_t_z_sb,
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
            new_point_x_y_sb,
            new_point_t_z_sb,
        ],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = input_size / workgroup_size / num_x_workgroups

    const shaderCode = genPreaggregationStage1ShaderCode(
        num_y_workgroups,
        workgroup_size,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [
                point_x_y_sb,
                point_t_z_sb,
                new_point_indices_sb,
                cluster_start_indices_sb,
                cluster_end_indices_sb,
                new_point_x_y_sb,
                new_point_t_z_sb,
            ],
        )

        const point_x_y = u8s_to_bigints(data[0], num_words, word_size)
        const point_t_z = u8s_to_bigints(data[1], num_words, word_size)
        const new_point_indices = u8s_to_numbers(data[2])
        const cluster_start_indices = u8s_to_numbers(data[3])
        const cluster_end_indices = u8s_to_numbers(data[4])
        const new_point_x_y = u8s_to_bigints(data[5], num_words, word_size)
        const new_point_t_z = u8s_to_bigints(data[6], num_words, word_size)

        verify_preagg_stage_1(
            point_x_y,
            point_t_z,
            new_point_indices,
            cluster_start_indices,
            cluster_end_indices,
            new_point_x_y,
            new_point_t_z,
        )
    }

    return { new_point_x_y_sb, new_point_t_z_sb }
}

const verify_preagg_stage_1 = (
    point_x_y: bigint[],
    point_t_z: bigint[],
    new_point_indices: number[],
    cluster_start_indices: number[],
    cluster_end_indices: number[],
    new_point_x_y: bigint[],
    new_point_t_z: bigint[],
) => {
    assert(point_x_y.length === point_t_z.length)
    assert(new_point_x_y.length === new_point_t_z.length)
    assert(cluster_start_indices.length === cluster_end_indices.length)
    assert(new_point_indices.length === cluster_end_indices.length)

    const points = construct_points(point_x_y, point_t_z)

    const expected: ExtPointType[] = []
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start = cluster_start_indices[i]
        const end = cluster_end_indices[i]
        let acc = points[new_point_indices[start]]
        for (let j = start + 1; j < end; j ++) {
            acc = acc.add(points[new_point_indices[j]])
        }
        expected.push(acc)
    }

    const new_points = construct_points(new_point_x_y, new_point_t_z)
    for (let i = 0; i < expected.length; i ++) {
        const n = new_points[i].toAffine()
        const m = expected[i].toAffine()
        assert(n.x === m.x && n.y === m.y, `mismatch at ${i}`)
    }
}

const genPreaggregationStage1ShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
) => {
    const misc_params = compute_misc_params(p, word_size)
    const num_words = misc_params.num_words
    const n0 = misc_params.n0
    const mask = BigInt(2) ** BigInt(word_size) - BigInt(1)
    const r = misc_params.r
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const mu_limbs = gen_mu_limbs(p, num_words, word_size)
    const p_bitlength = p.toString(2).length
    const slack = num_words * word_size - p_bitlength

    const shaderCode = mustache.render(
        preaggregation_stage_1_shader,
        {
            num_y_workgroups,
            workgroup_size,
            word_size,
            num_words,
            n0,
            p_limbs,
            r_limbs,
            mu_limbs,
            w_mask: (1 << word_size) - 1,
            slack,
            num_words_mul_two: num_words * 2,
            num_words_plus_one: num_words + 1,
            mask,
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
        },
        {
            structs,
            bigint_funcs,
            field_funcs,
            ec_funcs,
            montgomery_product_funcs,
        },
    )
    return shaderCode
}

export const pre_aggregation_stage_2_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    input_size: number,
    scalar_chunks_sb: GPUBuffer,
    cluster_start_indices_sb: GPUBuffer,
    new_point_indices_sb: GPUBuffer,
    debug = false,
): Promise<GPUBuffer> => {
    const new_scalar_chunks_sb = create_sb(device, input_size *  num_words * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
        ],
    )

    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            scalar_chunks_sb,
            new_point_indices_sb,
            cluster_start_indices_sb,
            new_scalar_chunks_sb,
        ],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = input_size / workgroup_size / num_x_workgroups

    const shaderCode = genPreaggregationStage2ShaderCode(
        num_y_workgroups,
        workgroup_size,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);
    
    if (debug) {
        // TODO
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [new_scalar_chunks_sb],
        )
        const nums = data.map(u8s_to_numbers_32)
        console.log(nums)
    }

    return new_scalar_chunks_sb
}

const genPreaggregationStage2ShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
) => {
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const mu_limbs = gen_mu_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        preaggregation_stage_2_shader,
        {
            num_y_workgroups,
            workgroup_size,
        },
        {
        },
    )
    return shaderCode
}

const compute_row_ptr = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    input_size: number,
    num_subtasks: number,
    num_rows_per_subtask: number,
    new_point_indices_sb: GPUBuffer,
    debug = false,
) => {
    /*
    const test_new_point_indices = [0, 2, 1, 3, 4, 5, 6, 0]
    new_point_indices_sb = create_and_write_sb(device, numbers_to_u8s_for_gpu(test_new_point_indices))
    input_size = test_new_point_indices.length
    num_subtasks = 1
    num_rows_per_subtask = 4
    */

    const row_ptr_sb = create_sb(device, (num_rows_per_subtask + 1) * 4)
    const num_chunks = input_size / num_subtasks
    const max_row_size = num_chunks / num_rows_per_subtask

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'storage',
        ],
    )

    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            new_point_indices_sb,
            row_ptr_sb,
        ],
    )

    const workgroup_size = 1
    const num_x_workgroups = 1
    const num_y_workgroups = 1

    const shaderCode = genComputeRowPtrShaderCode(
        num_y_workgroups,
        workgroup_size,
        num_chunks,
        max_row_size,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [ new_point_indices_sb, row_ptr_sb ],
        )
        
        const new_point_indices = u8s_to_numbers(data[0])
        const row_ptr = u8s_to_numbers(data[1])

        // Verify
        const expected: number[] = [0]
        for (let i = 0; i < new_point_indices.length; i += max_row_size) {
            let j = 0
            if (i === 0) {
                j = 1
            }
            for (; j < max_row_size; j ++) {
                if (new_point_indices[i + j] === 0) {
                    break
                }
            }
            expected.push(expected[expected.length - 1] + j)
        }
        assert(row_ptr.toString() === expected.toString(), 'row_ptr mismatch')
    }
}

const genComputeRowPtrShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
    num_chunks: number,
    max_row_size: number,
) => {
    const shaderCode = mustache.render(
        compute_row_ptr_shader,
        {
            num_y_workgroups,
            workgroup_size,
            num_chunks,
            max_row_size,
        },
        {
        },
    )
    return shaderCode
}

const construct_points = (x_y_coords: bigint[], t_z_coords: bigint[]) => {
    const points: ExtPointType[] = []
    for (let i = 0; i < x_y_coords.length; i += 2) {
        const pt = fieldMath.createPoint(
            fieldMath.Fp.mul(x_y_coords[i], rinv),
            fieldMath.Fp.mul(x_y_coords[i + 1], rinv),
            fieldMath.Fp.mul(t_z_coords[i], rinv),
            fieldMath.Fp.mul(t_z_coords[i + 1], rinv),
        )
        pt.assertValidity()
        points.push(pt)
    }
    return points
}
