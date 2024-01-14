import assert from 'assert'
import { BigIntPoint } from "../../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { ShaderManager } from '../shader_manager'
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

const p = BigInt('8444461749428370424248824938781546531375899335154063827935233455917409239041')
const word_size = 13
const params = compute_misc_params(p, word_size)
const num_words = params.num_words
const r = params.r
const rinv = params.rinv

import { FieldMath } from "../../reference/utils/FieldMath"
const fieldMath = new FieldMath()

/*
 * End-to-end implementation of the modified cuZK MSM algorithm by Lu et al,
 * 2022: https://eprint.iacr.org/2022/1321.pdf
 * Many aspects of cuZK were adapted and modified for our submission, and some
 * aspects were omitted. As such, please refer to the documentation we have
 * written for a more accurate description of our work. We also used techniques
 * by previous ZPrize contestations. In summary, we took the following
 * approach:
 * 1. Perform as much of the computation within the GPU as possible, in order
 *    to minimse CPU-GPU and GPU-CPU data transfer, which is slow.
 * 2. Use optimisations inspired by previous years' submissions, such as:
 *    - Montgomery multiplication with smaller limb sizes
 *    - Signed bucket indices
 * 3. Careful memory management to stay within WebGPU's default buffer size
 *    limits.
 * 4. A recursive bucket aggregation shader (a tree-summation method).
 * 5. Perform the final computation of the MSM result from the subtask results
 *    (Horner's rule) in the CPU instead of the GPU, as the number of points is
 *    small, and the time taken to compile a shader to perform this computation
 *    is greater than the time it takes for the CPU to do so.
 */
export const cuzk_gpu = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    log_result = true,
    force_recompile = false,
): Promise<{x: bigint, y: bigint}> => {
    const input_size = baseAffinePoints.length
    const chunk_size = input_size >= 65536 ? 16 : 4

    const shaderManager = new ShaderManager(
        word_size,
        chunk_size,
        input_size,
        force_recompile,
    )

    const num_columns = 2 ** chunk_size
    const num_rows = Math.ceil(input_size / num_columns)

    const num_chunks_per_scalar = Math.ceil(256 / chunk_size)
    const num_subtasks = num_chunks_per_scalar

    // Each pass must use the same GPUDevice and GPUCommandEncoder, or else
    // storage buffers can't be reused across compute passes
    const device = await get_device()
 
    // Convert the affine points to Montgomery form and decompose the scalars
    // using a single shader

    let c_workgroup_size = 64
    let c_num_x_workgroups = 128
    let c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups

    if (input_size <= 256) {
        c_workgroup_size = input_size
        c_num_x_workgroups = 1
        c_num_y_workgroups = 1
    } else if (input_size > 256 && input_size <= 32768) {
        c_workgroup_size = 64
        c_num_x_workgroups = 4
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    } else if (input_size > 32768 && input_size <= 65536) {
        c_workgroup_size = 64
        c_num_x_workgroups = 32
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    } else if (input_size > 65536 && input_size <= 131072) {
        c_workgroup_size = 64
        c_num_x_workgroups = 64
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    } else if (input_size > 131072 && input_size <= 262144) {
        c_workgroup_size = 64
        c_num_x_workgroups = 128
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    } else if (input_size > 262144 && input_size <= 524288) {
        c_workgroup_size = 64
        c_num_x_workgroups = 128
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    } else if (input_size > 524288 && input_size <= 1048576) {
        c_workgroup_size = 64
        c_num_x_workgroups = 128
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups
    }

    const c_shader = shaderManager.gen_convert_points_and_decomp_scalars_shader(
        c_workgroup_size,
        c_num_y_workgroups,
        num_subtasks,
        num_columns,
    )
    const { point_x_sb, point_y_sb, scalar_chunks_sb } =
        await convert_point_coords_and_decompose_shaders(
            c_shader,
            c_num_x_workgroups,
            c_num_y_workgroups,
            device,
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
    const bucket_sum_coord_bytelength = (num_columns / 2 + 1) * num_words * 4 * num_subtasks
    const bucket_sum_x_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_y_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_t_sb = create_sb(device, bucket_sum_coord_bytelength)
    const bucket_sum_z_sb = create_sb(device, bucket_sum_coord_bytelength)

    // Used by the parallel tree summation method in bucket_points_reduction
    //
    // Storage analyis:
    //      bucket_sum_coord_bytelength size: 41,944,320 bytes (41,944,320 / 80 = 524,304 bytes per coordinates required output buffer size)
    //      Max buffer size is: 268,435,456 bytes
    //      Max binding size: 134,217,728 bytes 
    //      Required buffer size per coordinate: bucket_sum_coord_bytelength / 2 == 20,972,160 bytes per coordinate < 134,217,728 bytes
    //      Representing 2^16 points = 4 coordinates * 20,972,160 bytes per coordinate (equaling 20,972,160 / 80 / 4 = 2^15 curve points) = 83,888,640 bytes (~83 MB)
    //      
    //      This means we can do the bucket aggregation in a single step for < 2^20 points. 
    const out_x_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_y_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_t_sb = create_sb(device, bucket_sum_coord_bytelength / 2)
    const out_z_sb = create_sb(device, bucket_sum_coord_bytelength / 2)

    const t_shader = shaderManager.gen_transpose_shader(num_subtasks)
    // Create single command encoder for device
    const commandEncoder = device.createCommandEncoder()

    // Transpose
    const {
        all_csc_col_ptr_sb,
        all_csc_val_idxs_sb,
    } = await transpose_gpu(
        t_shader,
        device,
        commandEncoder,
        input_size,
        num_columns,
        num_rows,
        num_subtasks,    
        scalar_chunks_sb,
    )

    const half_num_columns = num_columns / 2
    let s_workgroup_size = 64
    let s_num_x_workgroups = 64
    let s_num_y_workgroups = (half_num_columns / s_workgroup_size / s_num_x_workgroups)
    let s_num_z_workgroups = num_subtasks

    if (half_num_columns < 32768) {
        s_workgroup_size = 32
        s_num_x_workgroups = 1
        s_num_y_workgroups = Math.ceil(half_num_columns / s_workgroup_size / s_num_x_workgroups)
    }

    if (num_columns < 256) {
        s_workgroup_size = 1
        s_num_x_workgroups = half_num_columns
        s_num_y_workgroups = 1
        s_num_z_workgroups = 1
    }

    const smvp_shader = shaderManager.gen_smvp_shader(
        s_workgroup_size,
        s_num_y_workgroups,
        s_num_z_workgroups,
        num_columns,
    )

    // SMVP and multiplication by the bucket index
    await smvp_gpu(
        smvp_shader,
        s_num_x_workgroups,
        s_num_y_workgroups,
        s_num_z_workgroups,
        device,
        commandEncoder,
        num_subtasks,
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
    )

    const b_workgroup_size = 32
    const bucket_reduction_shader = shaderManager.gen_bucket_reduction_shader(
        b_workgroup_size,
    )

    // Bucket aggregation
    await bucket_aggregation(
        bucket_reduction_shader,
        b_workgroup_size,
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
        num_subtasks,
    )

    // Perform round of copying 
    commandEncoder.copyBufferToBuffer(
        out_x_sb,
        0,
        subtask_sum_x_sb,
        0,
        num_subtasks * num_words * 4,
    )
    commandEncoder.copyBufferToBuffer(
        out_y_sb,
        0,
        subtask_sum_y_sb,
        0,
        num_subtasks * num_words * 4,
    )
    commandEncoder.copyBufferToBuffer(
        out_t_sb,
        0,
        subtask_sum_t_sb,
        0,
        num_subtasks * num_words * 4,
    )
    commandEncoder.copyBufferToBuffer(
        out_z_sb,
        0,
        subtask_sum_z_sb,
        0,
        num_subtasks * num_words * 4,
    )

    // Read the subtask sums from the GPU
    const start = Date.now()
    const subtask_sum_data = await read_from_gpu(
        device,
        commandEncoder,
        [ subtask_sum_x_sb, subtask_sum_y_sb, subtask_sum_t_sb, subtask_sum_z_sb ],
    )

    // Destroy the GPU device object
    device.destroy()

    const x_mont_coords = u8s_to_bigints(subtask_sum_data[0], num_words, word_size)
    const y_mont_coords = u8s_to_bigints(subtask_sum_data[1], num_words, word_size)
    const t_mont_coords = u8s_to_bigints(subtask_sum_data[2], num_words, word_size)
    const z_mont_coords = u8s_to_bigints(subtask_sum_data[3], num_words, word_size)

    // Convert each point out of Montgomery form by multiplying by the inverse
    // of the Montgomery radix
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

    // Calculate the final result (Formula 3 of the cuZK paper, also known as
    // Horner's rule)
    const m = BigInt(2) ** BigInt(chunk_size)
    // The last scalar chunk is the most significant digit (base m)
    let result = points[points.length - 1]
    for (let i = points.length - 2; i >= 0; i --) {
        result = result.multiply(m)
        result = result.add(points[i])
    }
    const elapsed = Date.now() - start
    console.log(`Final steps (reading subtask sums, conversion out of Montgomery form, and Horner's rule) took ${elapsed}ms`)

    if (log_result) {
        console.log(result.toAffine())
    }

    return result.toAffine()
    // return { x: BigInt(0), y: BigInt(1) }
}

/*
 * Convert the affine points to Montgomery form, and decompose scalars into
 * chunk_size windows using the signed bucket index technique.

 * ASSUMPTION: the vast majority of WebGPU-enabled consumer devices have a
 * maximum buffer size of at least 268435456 bytes.
 * 
 * The default maximum buffer size is 268435456 bytes. Since each point
 * consumes 320 bytes, a maximum of around 2 ** 19 points can be stored in a
 * single buffer. If, however, we use 2 buffers - one for each point coordinate
 * X and Y - we can support larger input sizes.
 * Our implementation, however, will only support up to 2 ** 20 points as that
 * is the maximum input size for the ZPrize competition.
 *
 * Furthremore, there is a limit of 8 storage buffers per shader. As such, we
 * do not calculate the T and Z coordinates in this shader. Rather, we do so in
 * the SMVP shader.
 * 
 * Note that The test harness readme at
 * https://github.com/demox-labs/webgpu-msm states: "The submission should
 * produce correct outputs on input vectors with length up to 2^20. The
 * evaluation will be using input randomly sampled from size 2^16 ~ 2^20."
*/
export const convert_point_coords_and_decompose_shaders = async (
    shaderCode: string,
    num_x_workgroups: number,
    num_y_workgroups: number,
    device: GPUDevice,
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

    // Uniform param buffer
    const params_bytes = numbers_to_u8s_for_gpu(
        [input_size],
    )
    const params_ub = create_and_write_ub(device, params_bytes)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
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
            x_coords_sb,
            y_coords_sb,
            scalars_sb,
            point_x_sb,
            point_y_sb,
            scalar_chunks_sb,
            params_ub,
        ],
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const commandEncoder = device.createCommandEncoder()

    const start = Date.now()
    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1)
    device.queue.submit([commandEncoder.finish()])
    await device.queue.onSubmittedWorkDone()
    const elapsed = Date.now() - start
    console.log(`convert_point_coords_and_decompose_scalars took ${elapsed}ms`)

    // Debug the output of the shader. This should **not** be run in
    // production.
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

/*
 * Perform a modified version of CSR matrix transposition, which comes before
 * SMVP. Essentially, this step generates the point indices for each thread in
 * the SMVP step which corresponds to a particular bucket.
 */
export const transpose_gpu = async (
    shaderCode: string,
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

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )
    
    const start = Date.now()
    
    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1)

    // Debug the output of the shader. This should **not** be run in
    // production.
    const elapsed = Date.now() - start
    console.log(`transpose took ${elapsed}ms`)

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

        // Verify the output of the shader
        const expected = cpu_transpose(
            new_scalar_chunks,
            num_columns,
            num_rows,
            num_subtasks,
            input_size,
        )

        assert(expected.all_csc_col_ptr.toString() === all_csc_col_ptr_result.toString(), 'all_csc_col_ptr mismatch')
        assert(expected.all_csc_vals.toString() === all_csc_val_idxs_result.toString(), 'all_csc_vals mismatch')
    }

    return {
        all_csc_col_ptr_sb,
        all_csc_val_idxs_sb,
    }
}

/*
 * Compute the bucket sums and perform scalar multiplication with the bucket
 * indices.
 */
export const smvp_gpu = async (
    shaderCode: string,
    num_x_workgroups: number,
    num_y_workgroups: number,
    num_z_workgroups: number,
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    num_subtasks: number,
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
        [input_size],
    )
    const params_ub = create_and_write_ub(device, params_bytes)

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

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, num_z_workgroups)

    // Debug the output of the shader. This should **not** be run in
    // production.
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

        // Assertion checks take a long time!
        for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx++) {
            // Convert GPU output out of Montgomery coordinates
            const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
                return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z)
            }
            const output_points_gpu: ExtPointType[] = []
            for (let i = subtask_idx * (num_csr_cols / 2); i < subtask_idx * (num_csr_cols / 2) + (num_csr_cols / 2); i++) {
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
    }
    
    return {
        bucket_sum_x_sb,
        bucket_sum_y_sb,
        bucket_sum_t_sb,
        bucket_sum_z_sb,
    }
}

/*
 * Add up all the buckets (which have already been multiplied by their bucket
 * indices) using a recursive tree-summation method.
 */
export const bucket_aggregation = async (
    shaderCode: string,
    workgroup_size: number,
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
    num_subtasks: number,
    debug = false,
) => {
    const params = compute_misc_params(p, word_size)
    const num_words = params.num_words

    let original_bucket_sum_x_sb
    let original_bucket_sum_y_sb
    let original_bucket_sum_t_sb
    let original_bucket_sum_z_sb

    // Debug the output of the shader. This should **not** be run in
    // production.
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
            num_subtasks,
        )

        const e = s
        s = Math.ceil(s / 2)
        if (e === 1 && s === 1) {
            break
        }
    }
    
    // Debug the output of the shader. This should **not** be run in
    // production.
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

        for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx++) {
            // Convert the resulting point coordiantes out of Montgomery form
            const result = fieldMath.createPoint(
                fieldMath.Fp.mul(x_mont_coords_result[subtask_idx], rinv),
                fieldMath.Fp.mul(y_mont_coords_result[subtask_idx], rinv),
                fieldMath.Fp.mul(t_mont_coords_result[subtask_idx], rinv),
                fieldMath.Fp.mul(z_mont_coords_result[subtask_idx], rinv),
            )

            // Check that the sum of the points is correct
            const bucket_x_mont = u8s_to_bigints(data[4], num_words, word_size)
            const bucket_y_mont = u8s_to_bigints(data[5], num_words, word_size)
            const bucket_t_mont = u8s_to_bigints(data[6], num_words, word_size)
            const bucket_z_mont = u8s_to_bigints(data[7], num_words, word_size)

            const points: ExtPointType[] = []
            for (let i = subtask_idx * num_columns; i < subtask_idx * num_columns + num_columns; i++) {
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
    }

    return {
        out_x_sb,
        out_y_sb,
        out_t_sb,
        out_z_sb,
    }
}
