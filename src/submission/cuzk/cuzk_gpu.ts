import mustache from 'mustache'
import assert from 'assert'
import { BigIntPoint } from "../../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import {
    get_device,
    create_and_write_sb,
    create_and_write_ub,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
    execute_pipeline,
} from '../gpu'
import {
    gen_p_limbs,
    gen_r_limbs,
    gen_mu_limbs,
    u8s_to_bigints,
    u8s_to_numbers,
    u8s_to_numbers_32,
    bigints_to_u8_for_gpu,
    numbers_to_u8s_for_gpu,
    compute_misc_params,
    decompose_scalars_signed,
    are_point_arr_equal,
} from '../utils'
import { cpu_transpose } from './transpose'
import { cpu_smvp_signed } from './smvp';
import { shader_invocation } from '../bucket_points_reduction'

import convert_point_coords_and_decompose_scalars from '../wgsl/convert_point_coords_and_decompose_scalars.template.wgsl'
import extract_word_from_bytes_le_funcs from '../wgsl/extract_word_from_bytes_le.template.wgsl'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_funcs from '../wgsl/bigint/bigint.template.wgsl'
import field_funcs from '../wgsl/field/field.template.wgsl'
import ec_funcs from '../wgsl/curve/ec.template.wgsl'
import barrett_funcs from '../wgsl/barrett.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery/mont_pro_product.template.wgsl'
import curve_parameters from '../wgsl/curve/parameters.template.wgsl'
import transpose_serial_shader from '../wgsl/transpose_serial.wgsl'
import smvp_shader from '../wgsl/smvp.template.wgsl'
import bucket_points_reduction_shader from '../wgsl/bucket_points_reduction.template.wgsl'

// Hardcode params for word_size = 13
const p = BigInt('8444461749428370424248824938781546531375899335154063827935233455917409239041')
const word_size = 13
const params = compute_misc_params(p, word_size)
const n0 = params.n0
const num_words = params.num_words
const r = params.r
const rinv = params.rinv

import { FieldMath } from "../../reference/utils/FieldMath"
const fieldMath = new FieldMath()

/*
 * End-to-end implementation of the cuZK MSM algorithm.
 */
export const cuzk_gpu = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const input_size = baseAffinePoints.length
    const chunk_size = input_size >= 65536 ? 16 : 4

    const num_columns = 2 ** chunk_size
    const num_rows = Math.ceil(input_size / num_columns)

    const num_chunks_per_scalar = Math.ceil(256 / chunk_size)
    const num_subtasks = num_chunks_per_scalar

    // Each pass must use the same GPUDevice and GPUCommandEncoder, or else
    // storage buffers can't be reused across compute passes
    const device = await get_device()
    const commandEncoder = device.createCommandEncoder()
 
    // Convert the affine points to Montgomery form and decompose the scalars
    // using a single shader
    const { point_x_sb, point_y_sb, scalar_chunks_sb } =
        await convert_point_coords_and_decompose_shaders(
            device,
            commandEncoder,
            baseAffinePoints,
            num_words, 
            word_size,
            scalars,
            num_subtasks,
            num_columns,
            chunk_size,
        )

    // Buffers to contain the sum of all the bucket sums per subtask
    const subtask_sum_coord_bytelength = num_subtasks * num_words * 4
    const subtask_sum_x_sb = create_sb(device, subtask_sum_coord_bytelength)
    const subtask_sum_y_sb = create_sb(device, subtask_sum_coord_bytelength)
    const subtask_sum_t_sb = create_sb(device, subtask_sum_coord_bytelength)
    const subtask_sum_z_sb = create_sb(device, subtask_sum_coord_bytelength)

    // Buffers to  store the SMVP result (the bucket sum). They are overwritten
    // per iteration
    const bucket_sum_coord_bytelength = (num_columns / 2 + 1) * num_words * 4
    const bucket_sum_x_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_y_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_t_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_z_sb = create_sb(device, bucket_sum_coord_bytelength)

    // Used by the tree summation method in bucket_points_reduction
    const out_x_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_y_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_t_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_z_sb = create_sb(device, bucket_sum_coord_bytelength / 2)

    // Transpose
    const {
        all_csc_col_ptr_sb,
        all_csc_val_idxs_sb,
    } = await transpose_gpu(
        device,
        commandEncoder,
        input_size,
        num_columns,
        num_rows,
        num_subtasks,    
        scalar_chunks_sb,
        //true,
    )
    //device.destroy()
    //return { x: BigInt(0), y: BigInt(1) }

    for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
        // SMVP and multiplication by the bucket index
        await smvp_gpu(
            device,
            commandEncoder,
            subtask_idx,
            num_columns,
            input_size,
            chunk_size,
            all_csc_col_ptr_sb,
            point_x_sb,
            point_y_sb,
            all_csc_val_idxs_sb,
            bucket_sum_x_sb,
            bucket_sum_y_sb,
            bucket_sum_t_sb,
            bucket_sum_z_sb,
            //true,
        )
        //device.destroy()
        //return { x: BigInt(0), y: BigInt(1) }

        // Bucket aggregation
        await bucket_aggregation(
            device,
            commandEncoder,
            out_x_sb,
            out_y_sb,
            out_t_sb,
            out_z_sb,
            bucket_sum_x_sb,
            bucket_sum_y_sb,
            bucket_sum_t_sb,
            bucket_sum_z_sb,
            num_columns / 2,
        )

        commandEncoder.copyBufferToBuffer(
            out_x_sb,
            0,
            subtask_sum_x_sb,
            subtask_idx * num_words * 4,
            num_words * 4,
        )
        commandEncoder.copyBufferToBuffer(
            out_y_sb,
            0,
            subtask_sum_y_sb,
            subtask_idx * num_words * 4,
            num_words * 4,
        )
        commandEncoder.copyBufferToBuffer(
            out_t_sb,
            0,
            subtask_sum_t_sb,
            subtask_idx * num_words * 4,
            num_words * 4,
        )
        commandEncoder.copyBufferToBuffer(
            out_z_sb,
            0,
            subtask_sum_z_sb,
            subtask_idx * num_words * 4,
            num_words * 4,
        )
    }

    const subtask_sum_data = await read_from_gpu(
        device,
        commandEncoder,
        [ subtask_sum_x_sb, subtask_sum_y_sb, subtask_sum_t_sb, subtask_sum_z_sb ],
    )
    device.destroy()

    const x_mont_coords = u8s_to_bigints(subtask_sum_data[0], num_words, word_size)
    const y_mont_coords = u8s_to_bigints(subtask_sum_data[1], num_words, word_size)
    const t_mont_coords = u8s_to_bigints(subtask_sum_data[2], num_words, word_size)
    const z_mont_coords = u8s_to_bigints(subtask_sum_data[3], num_words, word_size)

    // Convert each point out of Montgomery form
    const points: ExtPointType[] = []
    for (let i = 0; i < num_subtasks; i ++) {
        const pt = fieldMath.createPoint(
            fieldMath.Fp.mul(x_mont_coords[i], rinv),
            fieldMath.Fp.mul(y_mont_coords[i], rinv),
            fieldMath.Fp.mul(t_mont_coords[i], rinv),
            fieldMath.Fp.mul(z_mont_coords[i], rinv),
        )
        points.push(pt)
    }

    // Horner's rule
    const m = BigInt(2) ** BigInt(chunk_size)
    // The last scalar chunk is the most significant digit (base m)
    let result = points[points.length - 1]
    for (let i = points.length - 2; i >= 0; i --) {
        result = result.multiply(m)
        result = result.add(points[i])
    }

    console.log(result.toAffine())
    return result.toAffine()
    //device.destroy()
    //return { x: BigInt(0), y: BigInt(1) }
}

/*
 * Convert the affine points to Montgomery form, and decompose scalars into
 * chunk_size windows.

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
export const convert_point_coords_and_decompose_shaders = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    baseAffinePoints: BigIntPoint[],
    num_words: number,
    word_size: number,
    scalars: bigint[],
    num_subtasks: number,
    num_columns: number,
    chunk_size: number,
    debug = false,
) => {
    assert(num_subtasks * chunk_size === 256)
    const input_size = baseAffinePoints.length

    // An affine point only contains X and Y points.
    const x_coords = Array(input_size).fill(BigInt(0))
    const y_coords = Array(input_size).fill(BigInt(0))
    for (let i = 0; i < input_size; i ++) {
        x_coords[i] = baseAffinePoints[i].x
        y_coords[i] = baseAffinePoints[i].y
    }

    // Convert points to bytes (performs ~2x faster than
    // `bigints_to_16_bit_words_for_gpu`)
    const x_coords_bytes = bigints_to_u8_for_gpu(x_coords, 16, 16)
    const y_coords_bytes = bigints_to_u8_for_gpu(y_coords, 16, 16)

    // Convert scalars to bytes
    const scalars_bytes = bigints_to_u8_for_gpu(scalars, 16, 16)

    // Input buffers
    const x_coords_sb = create_and_write_sb(device, x_coords_bytes)
    const y_coords_sb = create_and_write_sb(device, y_coords_bytes)
    const scalars_sb = create_and_write_sb(device, scalars_bytes)

    // Output buffers
    const point_x_sb = create_sb(device, input_size * num_words * 4)
    const point_y_sb = create_sb(device, input_size * num_words * 4)
    const scalar_chunks_sb = create_sb(device, input_size * num_subtasks * 4)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
            'storage',
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            x_coords_sb,
            y_coords_sb,
            scalars_sb,
            point_x_sb,
            point_y_sb,
            scalar_chunks_sb,
        ],
    )

    let workgroup_size = 256
    let num_x_workgroups = 1
    let num_y_workgroups = input_size / workgroup_size / num_x_workgroups

    if (input_size < 256) {
        workgroup_size = input_size
        num_x_workgroups = 1
        num_y_workgroups = 1
    } else if (input_size >= 256 && input_size < 65536) {
        workgroup_size = 256
        num_x_workgroups = input_size / workgroup_size
        num_y_workgroups = input_size / workgroup_size / num_x_workgroups
    }

    const shaderCode = genConvertPointCoordsAndDecomposeScalarsShaderCode(
        workgroup_size,
        num_y_workgroups,
        num_subtasks,
        num_columns,
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
            [
                point_x_sb,
                point_y_sb,
                scalar_chunks_sb,
            ],
        )
        
        // Verify point coords
        const computed_x_coords = u8s_to_bigints(data[0], num_words, word_size)
        const computed_y_coords = u8s_to_bigints(data[1], num_words, word_size)

        for (let i = 0; i < input_size; i ++) {
            const expected_x = baseAffinePoints[i].x * r % p
            const expected_y = baseAffinePoints[i].y * r % p

            if (!(expected_x === computed_x_coords[i] && expected_y === computed_y_coords[i])) {
                console.log('mismatch at', i)
                break
            }
        }

        // Verify scalar chunks
        const computed_chunks = u8s_to_numbers(data[2])

        const expected = decompose_scalars_signed(scalars, num_subtasks, chunk_size)

        for (let j = 0; j < expected.length; j++) {
            let z = 0;
            for (let i = j * input_size; i < (j + 1) * input_size; i++) {
                if (computed_chunks[i] !== expected[j][z]) {
                    throw Error(`scalar decomp mismatch at ${i}`)
                }
                z ++
            }
        }
    }

    return { point_x_sb, point_y_sb, scalar_chunks_sb }
}

const genConvertPointCoordsAndDecomposeScalarsShaderCode = (
    workgroup_size: number,
    num_y_workgroups: number,
    num_subtasks: number,
    num_columns: number,
    chunk_size: number, 
    input_size: number,
) => {
    const mask = BigInt(2) ** BigInt(word_size) - BigInt(1)
    const two_pow_word_size = 2 ** word_size
    const two_pow_chunk_size = 2 ** chunk_size
    const index_shift = 2 ** (chunk_size - 1)
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const mu_limbs = gen_mu_limbs(p, num_words, word_size)
    const p_bitlength = p.toString(2).length
    const slack = num_words * word_size - p_bitlength
        const shaderCode = mustache.render(
        convert_point_coords_and_decompose_scalars,
        {
            workgroup_size,
            num_y_workgroups,
            num_words,
            word_size,
            n0,
            mask,
            two_pow_word_size,
            two_pow_chunk_size,
            index_shift,
            p_limbs,
            r_limbs,
            mu_limbs,
            w_mask: (1 << word_size) - 1,
            slack,
            num_words_mul_two: num_words * 2,
            num_words_plus_one: num_words + 1,
            num_subtasks,
            num_columns,
            chunk_size,
            input_size,
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

export const transpose_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    input_size: number,
    num_columns: number,
    num_rows: number,
    num_subtasks: number,
    scalar_chunks_sb: GPUBuffer,
    debug = false,
): Promise<{
    all_csc_col_ptr_sb: GPUBuffer,
    all_csc_val_idxs_sb: GPUBuffer,
}> => {
    /*
     * n = number of columns (before transposition)
     * m = number of rows (before transposition)
     * nnz = number of nonzero elements
     *
     * Given: 
     *   - csr_col_idx (nnz) (aka the new_scalar_chunks)
     *
     * Output the transpose of the above:
     *   - csc_col_ptr (m + 1)
     *      - The column index of each nonzero element
     *   - csc_val_idxs (nnz)
     *      - The new index of each nonzero element
     *
     * Not computed as it's not used:
     *   - csc_row_idx (nnz)
     *      - The cumulative sum of the number of nonzero elements per row
     */

    const all_csc_col_ptr_sb = create_sb(device, num_subtasks * (num_columns + 1) * 4)
    const all_csc_val_idxs_sb = create_sb(device, scalar_chunks_sb.size)
    const all_curr_sb = create_sb(device, num_subtasks * num_columns * 4)

    const params_bytes = numbers_to_u8s_for_gpu(
        [num_rows, num_columns, input_size],
    )
    const params_ub = create_and_write_ub(device, params_bytes)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'storage',
            'storage',
            'storage',
            'uniform',
        ],
    )

    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            scalar_chunks_sb,
            all_csc_col_ptr_sb,
            all_csc_val_idxs_sb,
            all_curr_sb,
            params_ub,
        ],
    )

    const num_x_workgroups = 1
    const num_y_workgroups = 1
    const num_workgroups = num_subtasks 

    const shaderCode = mustache.render(
        transpose_serial_shader,
        { num_workgroups },
        {},
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
            [
                all_csc_col_ptr_sb,
                all_csc_val_idxs_sb,
                scalar_chunks_sb,
            ],
        )
    
        const all_csc_col_ptr_result = u8s_to_numbers_32(data[0])
        const all_csc_val_idxs_result = u8s_to_numbers_32(data[1])
        const new_scalar_chunks = u8s_to_numbers_32(data[2])

        console.log(
            //'new_scalar_chunks:', new_scalar_chunks, 
            //'num_columns:', num_columns,
            'all_csc_col_ptr_result:', all_csc_col_ptr_result,
            //'csc_val_idxs_result:', csc_val_idxs_result,
        )

        // Verify the output of the shader
        const expected = cpu_transpose(
            new_scalar_chunks,
            num_columns,
            num_rows,
            num_subtasks,
            input_size,
        )
        console.log('expected.csc_col_ptr', expected.all_csc_col_ptr)
        //console.log('expected.csc_vals', expected.csc_row_idx)

        assert(expected.all_csc_col_ptr.toString() === all_csc_col_ptr_result.toString(), 'all_csc_col_ptr mismatch')
        assert(expected.all_csc_vals.toString() === all_csc_val_idxs_result.toString(), 'all_csc_vals mismatch')
    }

    return {
        all_csc_col_ptr_sb,
        all_csc_val_idxs_sb,
    }
}

export const smvp_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    subtask_idx: number,
    num_csr_cols: number,
    input_size: number,
    chunk_size: number,
    all_csc_col_ptr_sb: GPUBuffer,
    point_x_sb: GPUBuffer,
    point_y_sb: GPUBuffer,
    all_csc_val_idxs_sb: GPUBuffer,
    bucket_sum_x_sb: GPUBuffer,
    bucket_sum_y_sb: GPUBuffer,
    bucket_sum_t_sb: GPUBuffer,
    bucket_sum_z_sb: GPUBuffer,
    debug = false,
) => {
    const params_bytes = numbers_to_u8s_for_gpu(
        [subtask_idx, input_size],
    )
    const params_ub = create_and_write_ub(device, params_bytes)
    const half_num_columns = num_csr_cols / 2

    let workgroup_size = 128
    let num_x_workgroups = 256
    let num_y_workgroups = (half_num_columns / workgroup_size / num_x_workgroups)

    if (num_csr_cols < 256) {
        workgroup_size = 1
        num_x_workgroups = half_num_columns
        num_y_workgroups = 1
    }

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
            'storage',
            'storage',
            'uniform',
        ],
    )

    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            all_csc_col_ptr_sb,
            all_csc_val_idxs_sb,
            point_x_sb,
            point_y_sb,
            bucket_sum_x_sb,
            bucket_sum_y_sb,
            bucket_sum_t_sb,
            bucket_sum_z_sb,
            params_ub,
        ],
    )

    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const index_shift = 2 ** (chunk_size - 1)
    const shaderCode = mustache.render(
        smvp_shader,
        {
            word_size,
            num_words,
            n0,
            p_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
            index_shift,
            workgroup_size,
            num_y_workgroups,
            num_columns: num_csr_cols,
            half_num_columns,
        },
        {
            structs,
            bigint_funcs,
            montgomery_product_funcs,
            field_funcs,
            curve_parameters,
            ec_funcs,
        },
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
            [
                all_csc_col_ptr_sb,
                all_csc_val_idxs_sb,
                point_x_sb,
                point_y_sb,
                bucket_sum_x_sb,
                bucket_sum_y_sb,
                bucket_sum_t_sb,
                bucket_sum_z_sb,
            ],
        )
    
        const all_csc_col_ptr_sb_result = u8s_to_numbers_32(data[0])
        const all_csc_val_idxs_result = u8s_to_numbers_32(data[1])
        const point_x_sb_result = u8s_to_bigints(data[2], num_words, word_size)
        const point_y_sb_result = u8s_to_bigints(data[3], num_words, word_size)
        const bucket_sum_x_sb_result = u8s_to_bigints(data[4], num_words, word_size)
        const bucket_sum_y_sb_result = u8s_to_bigints(data[5], num_words, word_size)
        const bucket_sum_t_sb_result = u8s_to_bigints(data[6], num_words, word_size)
        const bucket_sum_z_sb_result = u8s_to_bigints(data[7], num_words, word_size)

        // Convert GPU output out of Montgomery coordinates
        const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
            return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z)
        }
        const output_points_gpu: ExtPointType[] = []
        for (let i = 0; i < num_csr_cols / 2 + 1; i++) {
            const non = {
                x: fieldMath.Fp.mul(bucket_sum_x_sb_result[i], rinv),
                y: fieldMath.Fp.mul(bucket_sum_y_sb_result[i], rinv),
                t: fieldMath.Fp.mul(bucket_sum_t_sb_result[i], rinv),
                z: fieldMath.Fp.mul(bucket_sum_z_sb_result[i], rinv),
            }
            output_points_gpu.push(bigIntPointToExtPointType(non))
        }

        // Convert CPU output out of Montgomery coordinates
        const output_points_cpu_out_of_mont: ExtPointType[] = []
        for (let i = 0; i < input_size; i++) {
            const x = fieldMath.Fp.mul(point_x_sb_result[i], rinv)
            const y = fieldMath.Fp.mul(point_y_sb_result[i], rinv)
            const t = fieldMath.Fp.mul(x, y)
            const pt = fieldMath.createPoint(x, y, t, BigInt(1))
            pt.assertValidity()
            output_points_cpu_out_of_mont.push(pt)
        }

        // Calculate SMVP in CPU 
        const output_points_cpu: ExtPointType[] = cpu_smvp_signed(
            subtask_idx,
            input_size,
            num_csr_cols,
            chunk_size,
            all_csc_col_ptr_sb_result,
            all_csc_val_idxs_result,
            output_points_cpu_out_of_mont,
            fieldMath,
        )

        // Transform results into affine representation
        const output_points_affine_cpu = output_points_cpu.map((x) => x.toAffine())
        const output_points_affine_gpu = output_points_gpu.map((x) => x.toAffine())

        // Assert CPU and GPU output
        for (let i = 0; i < output_points_affine_gpu.length; i ++) {
            assert(output_points_affine_gpu[i].x === output_points_affine_cpu[i].x, "failed at i: " + i.toString())
            assert(output_points_affine_gpu[i].y === output_points_affine_cpu[i].y, "failed at i: " + i.toString())
        }
    }

    return {
        bucket_sum_x_sb,
        bucket_sum_y_sb,
        bucket_sum_t_sb,
        bucket_sum_z_sb,
    }
}

export const bucket_aggregation = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    out_x_sb: GPUBuffer,
    out_y_sb: GPUBuffer,
    out_t_sb: GPUBuffer,
    out_z_sb: GPUBuffer,
    bucket_sum_x_sb: GPUBuffer,
    bucket_sum_y_sb: GPUBuffer,
    bucket_sum_t_sb: GPUBuffer,
    bucket_sum_z_sb: GPUBuffer,
    num_columns: number,
    debug = false,
) => {
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    const p_limbs = gen_p_limbs(p, num_words, word_size)

    // Important: workgroup_size should be constant regardless of the number of
    // points, as setting a different workgroup_size will cause a costly
    // recompile. This constant is only passed into the shader as a template
    // variable for ease of benchmarking.
    const workgroup_size = 32

    const shaderCode = mustache.render(
        bucket_points_reduction_shader,
        {
            word_size,
            num_words,
            n0,
            p_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
            workgroup_size,
        },
        {
            structs,
            bigint_funcs,
            field_funcs,
            ec_funcs,
            curve_parameters,
            montgomery_product_funcs,
        },
    )

    let original_bucket_sum_x_sb
    let original_bucket_sum_y_sb
    let original_bucket_sum_t_sb
    let original_bucket_sum_z_sb

    if (debug) {
        original_bucket_sum_x_sb = create_sb(device, bucket_sum_x_sb.size)
        original_bucket_sum_y_sb = create_sb(device, bucket_sum_y_sb.size)
        original_bucket_sum_t_sb = create_sb(device, bucket_sum_t_sb.size)
        original_bucket_sum_z_sb = create_sb(device, bucket_sum_z_sb.size)

        commandEncoder.copyBufferToBuffer(
            bucket_sum_x_sb,
            0,
            original_bucket_sum_x_sb,
            0,
            bucket_sum_x_sb.size,
        )
        commandEncoder.copyBufferToBuffer(
            bucket_sum_y_sb,
            0,
            original_bucket_sum_y_sb,
            0,
            bucket_sum_y_sb.size,
        )
        commandEncoder.copyBufferToBuffer(
            bucket_sum_t_sb,
            0,
            original_bucket_sum_t_sb,
            0,
            bucket_sum_t_sb.size,
        )
        commandEncoder.copyBufferToBuffer(
            bucket_sum_z_sb,
            0,
            original_bucket_sum_z_sb,
            0,
            bucket_sum_z_sb.size,
        )
    }

    //let num_invocations = 0
    let s = num_columns
    while (s > 1) {
        await shader_invocation(
            device,
            commandEncoder,
            shaderCode,
            bucket_sum_x_sb,
            bucket_sum_y_sb,
            bucket_sum_t_sb,
            bucket_sum_z_sb,
            out_x_sb,
            out_y_sb,
            out_t_sb,
            out_z_sb,
            s,
            num_words,
            workgroup_size,
        )
        //num_invocations ++

        const e = s
        s = Math.ceil(s / 2)
        if (e === 1 && s === 1) {
            break
        }
    }
    
    if (
        debug
        && original_bucket_sum_x_sb != undefined // prevent TS warnings
        && original_bucket_sum_y_sb != undefined
        && original_bucket_sum_t_sb != undefined
        && original_bucket_sum_z_sb != undefined
    ) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [
                out_x_sb,
                out_y_sb,
                out_t_sb,
                out_z_sb,
                original_bucket_sum_x_sb,
                original_bucket_sum_y_sb,
                original_bucket_sum_t_sb,
                original_bucket_sum_z_sb,
            ]
        )

        const x_mont_coords_result = u8s_to_bigints(data[0], num_words, word_size)
        const y_mont_coords_result = u8s_to_bigints(data[1], num_words, word_size)
        const t_mont_coords_result = u8s_to_bigints(data[2], num_words, word_size)
        const z_mont_coords_result = u8s_to_bigints(data[3], num_words, word_size)

        // Convert the resulting point coordiantes out of Montgomery form
        const result = fieldMath.createPoint(
            fieldMath.Fp.mul(x_mont_coords_result[0], rinv),
            fieldMath.Fp.mul(y_mont_coords_result[0], rinv),
            fieldMath.Fp.mul(t_mont_coords_result[0], rinv),
            fieldMath.Fp.mul(z_mont_coords_result[0], rinv),
        )

        // Check that the sum of the points is correct
        const bucket_x_mont = u8s_to_bigints(data[4], num_words, word_size)
        const bucket_y_mont = u8s_to_bigints(data[5], num_words, word_size)
        const bucket_t_mont = u8s_to_bigints(data[6], num_words, word_size)
        const bucket_z_mont = u8s_to_bigints(data[7], num_words, word_size)

        const points: ExtPointType[] = []
        for (let i = 0; i < num_columns; i ++) {
            points.push(fieldMath.createPoint(
                fieldMath.Fp.mul(bucket_x_mont[i], rinv),
                fieldMath.Fp.mul(bucket_y_mont[i], rinv),
                fieldMath.Fp.mul(bucket_t_mont[i], rinv),
                fieldMath.Fp.mul(bucket_z_mont[i], rinv),
            ))
        }

        // Add up the original points
        let expected = points[0]
        for (let i = 1; i < points.length; i ++) {
            expected = expected.add(points[i])
        }

        assert(are_point_arr_equal([result], [expected]))
    }

    return {
        out_x_sb,
        out_y_sb,
        out_t_sb,
        out_z_sb,
    }
}
