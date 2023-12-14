import assert from 'assert'
import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import { FieldMath } from "../reference/utils/FieldMath";
import {
    get_device,
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
    execute_pipeline,
} from './gpu'
import structs from './wgsl/struct/structs.template.wgsl'
import bigint_funcs from './wgsl/bigint/bigint.template.wgsl'
import field_funcs from './wgsl/field/field.template.wgsl'
import ec_funcs from './wgsl/curve/ec.template.wgsl'
import curve_parameters from './wgsl/curve/parameters.template.wgsl'
import montgomery_product_funcs from './wgsl/montgomery/mont_pro_product.template.wgsl'
import bucket_points_reduction_shader from './wgsl/bucket_points_reduction.template.wgsl'
import { are_point_arr_equal, compute_misc_params, u8s_to_bigints, numbers_to_u8s_for_gpu, gen_p_limbs, bigints_to_u8_for_gpu } from './utils'

export const bucket_points_reduction = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    await test_bucket_points_reduction(baseAffinePoints, baseAffinePoints.length)
    //for (let i = 2; i < 64; i ++) {
        //await test_bucket_points_reduction(baseAffinePoints, i)
    //}
    return { x: BigInt(0), y: BigInt(0) }
}

export const test_bucket_points_reduction = async (
    baseAffinePoints: BigIntPoint[],
    input_size: number,
) => {
    assert(baseAffinePoints.length >= input_size)

    const fieldMath = new FieldMath()
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const word_size = 13

    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    const r = params.r
    const rinv = params.rinv
    const p_limbs = gen_p_limbs(p, num_words, word_size)

    const x_y_coords: bigint[] = []
    const t_z_coords: bigint[] = []
    for (const pt of baseAffinePoints.slice(0, input_size)) {
        x_y_coords.push(fieldMath.Fp.mul(pt.x, r))
        x_y_coords.push(fieldMath.Fp.mul(pt.y, r))
        t_z_coords.push(fieldMath.Fp.mul(pt.t, r))
        t_z_coords.push(fieldMath.Fp.mul(pt.z, r))
    }

    const points = baseAffinePoints.slice(0, input_size).map((x) => fieldMath.createPoint(x.x, x.y, x.t, x.z))
    let expected = points[0]
    for (let i = 1; i < points.length; i ++) {
        expected  = expected.add(points[i])
    }

    const device = await get_device()
    const commandEncoder = device.createCommandEncoder()

    const x_y_coords_bytes = bigints_to_u8_for_gpu(x_y_coords, num_words, word_size)
    const t_z_coords_bytes = bigints_to_u8_for_gpu(t_z_coords, num_words, word_size)

    let x_y_coords_sb = create_and_write_sb(device, x_y_coords_bytes)
    let t_z_coords_sb = create_and_write_sb(device, t_z_coords_bytes)

    const shaderCode = mustache.render(
        bucket_points_reduction_shader,
        {
            word_size,
            num_words,
            n0,
            p_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
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

    let num_invocations = 0
    let s = input_size
    const start = Date.now()
    while (s > 1) {
        const r = await shader_invocation(
            device,
            commandEncoder,
            shaderCode,
            x_y_coords_sb,
            t_z_coords_sb,
            s,
            num_words,
            word_size,
        )
        num_invocations ++
        x_y_coords_sb = r.out_x_y_sb
        t_z_coords_sb = r.out_t_z_sb

        const e = s
        s = Math.ceil(s / 2)
        if (e === 1 && s === 1) {
            break
        }
    }
    const elapsed = Date.now() - start
    console.log(`${num_invocations} invocations of the point reduction shader for ${input_size} points took ${elapsed}ms`)

    const data = await read_from_gpu(
        device,
        commandEncoder,
        [ x_y_coords_sb, t_z_coords_sb ]
    )

    const x_y_mont_coords_result = u8s_to_bigints(data[0], num_words, word_size)
    const t_z_mont_coords_result = u8s_to_bigints(data[1], num_words, word_size)

    const result = fieldMath.createPoint(
        fieldMath.Fp.mul(x_y_mont_coords_result[0], rinv),
        fieldMath.Fp.mul(x_y_mont_coords_result[1], rinv),
        fieldMath.Fp.mul(t_z_mont_coords_result[0], rinv),
        fieldMath.Fp.mul(t_z_mont_coords_result[1], rinv),
    )

    //console.log('result:', result)
    //console.log('result.isAffine():', result.toAffine())
    //console.log('expected.isAffine():', expected.toAffine())
    assert(are_point_arr_equal([result], [expected]))

    device.destroy()
}

const shader_invocation = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    shaderCode: string,
    x_y_coords_sb: GPUBuffer,
    t_z_coords_sb: GPUBuffer,
    num_points: number,
    num_words: number,
    word_size: number,
) => {
    assert(num_points <= 2 ** 16)

    const num_points_bytes = numbers_to_u8s_for_gpu([num_points])
    const num_points_sb = create_and_write_sb(device, num_points_bytes)

    // Use only the right amount of memory one needs for the storage buffers
    const out_x_y_sb = create_sb(device, Math.ceil(num_points / 2) * 4 * num_words * word_size)
    const out_t_z_sb = create_sb(device, Math.ceil(num_points / 2) * 4 * num_words * word_size)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
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
        [ x_y_coords_sb, t_z_coords_sb, num_points_sb, out_x_y_sb, out_t_z_sb ],
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const num_x_workgroups = 256
    const num_y_workgroups = 256

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    return { out_x_y_sb, out_t_z_sb, num_points_sb }
}
