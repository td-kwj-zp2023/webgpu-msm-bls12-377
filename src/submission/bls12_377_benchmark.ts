import mustache from 'mustache'
import assert from 'assert'
import {
    gen_p_limbs,
    gen_r_limbs,
    u8s_to_bigints,
    compute_misc_params,
    bigints_to_u8_for_gpu,
} from './utils'
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
import bls12_377_add_points_shader from './wgsl/bls12_377_add_points_shader.template.wgsl'
import { BigIntPoint } from '../reference/types'
import { G1 } from '@celo/bls12377js'
import structs from './wgsl/struct/structs.template.wgsl'
import bigint_funcs from './wgsl/bigint/bigint.template.wgsl'
import field_funcs from './wgsl/field/field.template.wgsl'
import ec_funcs from './wgsl/curve/ec_bls1_377.template.wgsl'
import montgomery_product_funcs from './wgsl/montgomery/mont_pro_product.template.wgsl'
import {
    BASE_FIELD,
    createGeneratorPoint,
    createBaseF,
    createAffinePoint,
    get_bigint_x_y,
} from './bls12_377'


export const bls12_377_benchmark = async (
    {}: BigIntPoint[],
    {}: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const p = BASE_FIELD
    const word_size = 13
    const num_inputs = 512

    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    const r = params.r
    const rinv = params.rinv

    const points: G1[] = []
    const x_coords: bigint[] = []
    const y_coords: bigint[] = []

    const g = createGeneratorPoint()
    for (let i = 0; i < num_inputs; i ++) {
        const s = createBaseF(BigInt(i + 1))
        const pt = g.scalarMult(s.toBig()).toAffine()
        points.push(pt)

        const { x, y } = get_bigint_x_y(pt)
        x_coords.push(x)
        y_coords.push(y)
    }

    // Convert to Montgomery form
    const x_coords_mont: bigint[] = []
    const y_coords_mont: bigint[] = []
    for (let i = 0; i < num_inputs; i ++) {
        const x_mont = x_coords[i] * r % p
        const y_mont = y_coords[i] * r % p
        x_coords_mont.push(x_mont)
        y_coords_mont.push(y_mont)
    }

    const x_coords_mont_bytes = bigints_to_u8_for_gpu(x_coords_mont, num_words, word_size)
    const y_coords_mont_bytes = bigints_to_u8_for_gpu(y_coords_mont, num_words, word_size)

    const device = await get_device()
    const commandEncoder = device.createCommandEncoder()

    const x_coords_sb = create_and_write_sb(device, x_coords_mont_bytes)
    const y_coords_sb = create_and_write_sb(device, y_coords_mont_bytes)
    const out_x_coords_sb = create_sb(device, x_coords_mont_bytes.length)
    const out_y_coords_sb = create_sb(device, y_coords_mont_bytes.length)
    const out_z_coords_sb = create_sb(device, y_coords_mont_bytes.length)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
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
            out_x_coords_sb,
            out_y_coords_sb,
            out_z_coords_sb,
        ],
    )

    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const r_limbs = gen_r_limbs(r, num_words, word_size)
    const shaderCode = mustache.render(
        bls12_377_add_points_shader,
        {
            word_size,
            num_words,
            n0,
            p_limbs,
            r_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
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

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const num_x_workgroups = num_inputs / 2
    const num_y_workgroups = 1

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1)

    const data = await read_from_gpu(
        device,
        commandEncoder,
        [
            out_x_coords_sb,
            out_y_coords_sb,
            out_z_coords_sb,
        ],
    )
    device.destroy()

    const out_x_coords_mont = u8s_to_bigints(data[0], num_words, word_size)
    const out_y_coords_mont = u8s_to_bigints(data[1], num_words, word_size)
    const out_z_coords_mont = u8s_to_bigints(data[2], num_words, word_size)

    const out_x_coords: bigint[] = []
    const out_y_coords: bigint[] = []
    const out_z_coords: bigint[] = []

    // Convert out of Montgomery form
    for (let i = 0; i < num_inputs; i ++) {
        out_x_coords.push(out_x_coords_mont[i] * rinv % p)
        out_y_coords.push(out_y_coords_mont[i] * rinv % p)
        out_z_coords.push(out_z_coords_mont[i] * rinv % p)
    }

    for (let i = 0; i < num_inputs / 2; i ++) {
        const ai = i * 2
        const di = ai + 1
        const added_pt = createAffinePoint(
            out_x_coords[ai],
            out_y_coords[ai],
            out_z_coords[ai],
        )
        const added_coords = get_bigint_x_y(added_pt)
        const added_expected = points[ai].add(points[di]).toAffine()

        assert(added_coords.x.toString() === added_expected.x().toBig().toString(), `mismatch at ${ai}`)
        assert(added_coords.y.toString() === added_expected.y().toBig().toString())

        const doubled_pt = createAffinePoint(
            out_x_coords[di],
            out_y_coords[di],
            out_z_coords[di],
        )
        const doubled_coords = get_bigint_x_y(doubled_pt)
        const doubled_expected = added_expected.dbl().toAffine()

        assert(doubled_coords.x.toString() === doubled_expected.x().toBig().toString(), `mismatch at ${di}`)
        assert(doubled_coords.y.toString() === doubled_expected.y().toBig().toString())

    }
    console.log('add_points() and double_point() work')

    return { x: BigInt(1), y: BigInt(0) }
}
