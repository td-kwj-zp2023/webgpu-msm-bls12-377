import mustache from 'mustache'
import { BigIntPoint } from "../../reference/types"
import {
    get_device,
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
} from '../gpu'
import {
    to_words_le,
    gen_p_limbs,
    gen_r_limbs,
    gen_mu_limbs,
    u8s_to_bigints,
    u8s_to_numbers,
    u8s_to_numbers_32,
    bigints_to_16_bit_words_for_gpu,
} from '../utils'

/*
 * End-to-end implementation of the cuZK MSM algorithm.
 */
export const cuzk_gpu = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const input_size = scalars.length
    const num_subtasks = 16

    const device = await get_device()

    // Convert the affine points to Montgomery form in the GPU
    const { point_x_sb, point_y_sb, point_t_sb, point_z_sb } =
        await convert_point_coords_to_mont_gpu(device, baseAffinePoints, false)

    // Decompose the scalars
    const scalar_chunk_sb = await decompose_scalars_gpu(
        device,
        scalars,
        num_subtasks,
        false,
    )

    for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
        const { new_point_indices_sb, cluster_start_indices_sb, cluster_end_indices_sb } =
            await csr_precompute_gpu(
                device,
                input_size,
                subtask_idx,
                scalar_chunk_sb,
                false,
            )
    }
    
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
const convert_point_coords_to_mont_gpu = async (
    device: GPUDevice,
    baseAffinePoints: BigIntPoint[],
    debug = false,
): Promise<{
    point_x_sb: GPUBuffer, point_y_sb: GPUBuffer, point_t_sb: GPUBuffer, point_z_sb: GPUBuffer,
}> => {
    const input_size = baseAffinePoints.length

    // An affine point only contains X and Y points.
    const x_coords = baseAffinePoints.map((b) => b.x)
    const y_coords = baseAffinePoints.map((b) => b.y)

    const x_coords_bytes = bigints_to_16_bit_words_for_gpu(x_coords)
    const y_coords_bytes = bigints_to_16_bit_words_for_gpu(y_coords)

    // Input buffers
    const x_coords_sb = create_and_write_sb(device, x_coords_bytes)
    const y_coords_sb = create_and_write_sb(device, y_coords_bytes)

    // Output buffers
    const point_x_sb = create_sb(device, input_size * num_words * 4)
    const point_y_sb = create_sb(device, input_size * num_words * 4)
    const point_t_sb = create_sb(device, input_size * num_words * 4)
    const point_z_sb = create_sb(device, input_size * num_words * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
            'storage',
            'storage'
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            x_coords_sb,
            y_coords_sb,
            point_x_sb,
            point_y_sb,
            point_t_sb,
            point_z_sb,
        ],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = x_coords.length / workgroup_size / num_x_workgroups

    const shaderCode = genConvertPointCoordsShaderCode(
        num_y_workgroups,
        workgroup_size,
    )

    const computePipeline = create_compute_pipeline(
        device,
        bindGroupLayout,
        shaderCode,
        'main',
    )

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1)
    passEncoder.end()

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [point_x_sb, point_y_sb, point_t_sb, point_z_sb],
        )

        // Check point_x data
        const computed_x_coords = u8s_to_bigints(data[0], num_words, word_size)
        const computed_y_coords = u8s_to_bigints(data[1], num_words, word_size)
        const computed_t_coords = u8s_to_bigints(data[2], num_words, word_size)
        const computed_z_coords = u8s_to_bigints(data[3], num_words, word_size)

        for (let i = 0; i < input_size; i ++) {
            const expected_x = baseAffinePoints[i].x * r % p
            const expected_y = baseAffinePoints[i].y * r % p
            const expected_t = (baseAffinePoints[i].x * baseAffinePoints[i].y * r) % p
            const expected_z = r % p

            if (!(
                expected_x === computed_x_coords[i] &&
                expected_y === computed_y_coords[i] &&
                expected_t === computed_t_coords[i] &&
                expected_z === computed_z_coords[i]
            )) {
                console.log('mismatch at', i)
                break
            }
        }
    }

    return { point_x_sb, point_y_sb, point_t_sb, point_z_sb }
}

import convert_point_coords_shader from '../wgsl/convert_point_coords.template.wgsl'
import extract_word_from_bytes_le_funcs from '../wgsl/extract_word_from_bytes_le.template.wgsl'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_funcs from '../wgsl/bigint/bigint.template.wgsl'
import field_funcs from '../wgsl/field/field.template.wgsl'
import barrett_funcs from '../wgsl/barrett.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery/mont_pro_product.template.wgsl'

const p = BigInt('8444461749428370424248824938781546531375899335154063827935233455917409239041')
const r = BigInt('3336304672246003866643098545847835280997800251509046313505217280697450888997')
const word_size = 13
const num_words = 20
const n0 = 8191
const mask = 8191
const slack = 7

const genConvertPointCoordsShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
) => {
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const mu_limbs = gen_mu_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        convert_point_coords_shader,
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
            extract_word_from_bytes_le_funcs,
            structs,
            bigint_funcs,
            field_funcs,
            barrett_funcs,
            montgomery_product_funcs,
        },
    )
    return shaderCode
}

const decompose_scalars_gpu = async (
    device: GPUDevice,
    scalars: bigint[],
    num_subtasks: number,
    debug = false,
): Promise<GPUBuffer> => {
    const input_size = scalars.length
    const chunk_size = Math.ceil(256 / num_subtasks)

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
        num_y_workgroups,
        workgroup_size,
        num_subtasks,
        chunk_size
    )

    const computePipeline = create_compute_pipeline(
        device,
        bindGroupLayout,
        shaderCode,
        'main',
    )

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1)
    passEncoder.end()

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [chunks_sb],
        )

        const computed_chunks = u8s_to_numbers(data[0])
        const expected: number[] = []
        for (const scalar of scalars) {
            const chunks = to_words_le(scalar, num_subtasks, chunk_size)
            for (const chunk of chunks) {
                expected.push(chunk)
            }
        }

        if (computed_chunks.length !== expected.length) {
            throw Error('output size mismatch')
        }

        for (let i = 0; i < computed_chunks.length; i ++) {
            if (computed_chunks[i].toString() !== expected[i].toString()) {
                throw Error(`scalar decomp mismatch at ${i}`)
            }
        }
    }

    return chunks_sb
}

import decompose_scalars_shader from '../wgsl/decompose_scalars.template.wgsl'
const genDecomposeScalarsShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
    num_subtasks: number,
    chunk_size: number,
) => {
    const shaderCode = mustache.render(
        decompose_scalars_shader,
        {
            workgroup_size,
            num_y_workgroups,
            num_subtasks,
            chunk_size,
        },
        {
            extract_word_from_bytes_le_funcs,
        },
    )
    return shaderCode
}

const csr_precompute_gpu = async (
    device: GPUDevice,
    input_size: number,
    subtask_idx: number,
    scalar_chunk_sb: GPUBuffer,
    debug = false,
): Promise<{
    new_point_indices_sb: GPUBuffer,
    cluster_start_indices_sb: GPUBuffer,
    cluster_end_indices_sb: GPUBuffer,
}> => {
    // Output buffers
    const new_point_indices_sb = create_sb(device, input_size * 4)
    const cluster_start_indices_sb = create_sb(device, input_size * 4)
    const cluster_end_indices_sb = create_sb(device, input_size * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        ['storage', 'storage', 'storage']
    )

    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
        ],
    )

    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = input_size / workgroup_size / num_x_workgroups

    const shaderCode = genCsrPrecomputeShaderCode(
        num_y_workgroups,
        workgroup_size,
    )

    const computePipeline = create_compute_pipeline(
        device,
        bindGroupLayout,
        shaderCode,
        'main',
    )

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1)
    passEncoder.end()

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [
                new_point_indices_sb,
                cluster_start_indices_sb,
                cluster_end_indices_sb,
            ],
        )

        const nums = data.map(u8s_to_numbers_32)

        for (let i = 0; i < input_size; i ++) {
            if (nums[0][i] !== i) {
                throw Error(`new_point_indices_sb mismatch at ${i}`)
            }
            if (nums[1][i] !== i) {
                throw Error(`cluster_start_indices_sb mismatch at ${i}`)
            }

            if (nums[2][i] - 1 !== i) {
                throw Error(`cluster_end_indices_sb mismatch at ${i}`)
            }
        }
    }
    return { new_point_indices_sb, cluster_start_indices_sb, cluster_end_indices_sb }
}

import gen_csr_precompute_shader from '../wgsl/gen_csr_precompute.template.wgsl'
const genCsrPrecomputeShaderCode = (
    num_y_workgroups: number,
    workgroup_size: number,
) => {
    const shaderCode = mustache.render(
        gen_csr_precompute_shader,
        {
            workgroup_size,
            num_y_workgroups,
        },
        {
        },
    )
    return shaderCode
}
