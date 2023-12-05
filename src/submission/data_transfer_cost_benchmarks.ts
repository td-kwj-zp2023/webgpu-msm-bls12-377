import { BigIntPoint } from "../reference/types"
import mustache from 'mustache'
import {
    get_device,
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
} from './gpu'
import {
    numbers_to_u8s_for_gpu,
} from './utils'
import simple_shader from './wgsl/simple.wgsl'
import complex_shader from './wgsl/complex.wgsl'


/*
 * Benchmark data transfer costs
 */
export const data_transfer_cost_benchmarks = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {

    let num_bytes = 1 * 1024 * 1024

    console.log('Simple shader benchmarks:')
    await shader_benchmark(simple_shader, num_bytes, true)
    await shader_benchmark(simple_shader, num_bytes, false)

    console.log('Complex shader benchmarks:')
    await shader_benchmark(complex_shader, num_bytes, true)
    await shader_benchmark(complex_shader, num_bytes, false)

    num_bytes = 32 * 1024 * 1024

    console.log('Simple shader benchmarks:')
    await shader_benchmark(simple_shader, num_bytes, true)
    await shader_benchmark(simple_shader, num_bytes, false)

    console.log('Complex shader benchmarks:')
    await shader_benchmark(complex_shader, num_bytes, true)
    await shader_benchmark(complex_shader, num_bytes, false)

    return { x: BigInt(1), y: BigInt(0) }
}

const shader_benchmark = async (
    shaderCode: string,
    num_bytes: number,
    read: boolean,
) => {
    const device = await get_device()
    const commandEncoder = device.createCommandEncoder()

    const data = Array(num_bytes / 4).fill(2 ** 32 - 1)
    const data_bytes = numbers_to_u8s_for_gpu(data)

    const start = Date.now()

    const data_sb = create_and_write_sb(device, data_bytes)

    const bindGroupLayout = create_bind_group_layout(
        device,
        ['storage'],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [data_sb],
    )

    const workgroup_size = 1
    const num_x_workgroups = 1

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups)
    passEncoder.end()

    if (read) {
        await read_from_gpu(
            device,
            commandEncoder,
            [data_sb],
        )
    }
    const elapsed = Date.now() - start
    if (read) {
        console.log(`Writing to and reading from GPU ${num_bytes} bytes (${num_bytes / 1024 / 1024} MB) took ${elapsed} ms`)
    } else {
        console.log(`Writing to GPU (but not reading from) ${num_bytes} bytes (${num_bytes / 1024 / 1024} MB) took ${elapsed} ms`)
    }
}

/*
const convert_point_coords_to_mont_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    baseAffinePoints: BigIntPoint[],
    debug = true,
): Promise<any> => {
    const input_size = baseAffinePoints.length

    // An affine point only contains X and Y points.
    const x_y_coords = Array(input_size * 2).fill(BigInt(0))
    for (let i = 0; i < input_size; i ++) {
        x_y_coords[i * 2] = baseAffinePoints[i].x
        x_y_coords[i * 2 + 1] = baseAffinePoints[i].y
    }

    const x_y_coords_bytes = bigints_to_u8_for_gpu(x_y_coords, 20, 13)

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

    const workgroup_size = 256
    const num_x_workgroups = 256

    const shaderCode = genConvertPointCoordsShaderCode(
        workgroup_size,
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups)
    passEncoder.end()

    if (debug) {
        const data = await read_from_gpu(
            device,
            commandEncoder,
            [point_x_y_sb, point_t_z_sb],
        )

        // Check point_x data
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
                // debugger
                break
            }
        }
    }

    console.log("montgomery conversion assertion checks pass!")

    return { point_x_y_sb, point_t_z_sb }
}

const genConvertPointCoordsShaderCode = (
    workgroup_size: number,
) => {
    const p_bitlength = p.toString(2).length
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const misc_params = compute_misc_params(p, word_size)
    const m_limbs = gen_barrett_domb_m_limbs(misc_params.barrett_domb_m, num_words, word_size)
    const n0 = misc_params.n0
    const slack = num_words * word_size - p_bitlength
    const mask = BigInt(2) ** BigInt(word_size) - BigInt(1)
    const two_pow_word_size = 2 ** word_size

    const shaderCode = mustache.render(
        convert_point_coords_shader,
        {
            workgroup_size,
            num_words,
            word_size,
            n0,
            mask,
            two_pow_word_size,
            p_limbs,
            r_limbs,
            m_limbs,
            w_mask: (1 << word_size) - 1,
            slack,
            num_words_mul_two: num_words * 2,
            num_words_plus_one: num_words + 1,
            z: (word_size * num_words) - p_bitlength
        },
        {
            structs,
            bigint_funcs,
            field_funcs,
            barrett_functions,
            montgomery_product_funcs,
        },
    )
    return shaderCode
}
*/
