import mustache from 'mustache'
import assert from 'assert'
import { BigIntPoint } from "../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../reference/utils/FieldMath";
import { genRandomFieldElement } from './utils'
import { compute_misc_params, u8s_to_points, points_to_u8s_for_gpu, gen_p_limbs } from './utils'
import add_points_any_a_shader from '../submission/wgsl/add_points_any_a.template.wgsl'
import add_points_a_minus_one_shader from '../submission/wgsl/add_points_a_minus_one.template.wgsl'
import bigint_struct from '../submission/wgsl/structs/bigint.template.wgsl'
import bigint_funcs from '../submission/wgsl/bigint.template.wgsl'
import montgomery_product_funcs from '../submission/wgsl/montgomery_product.template.wgsl'
import { get_device, create_bind_group } from './gpu'
import { add_points_any_a } from './add_points'

const setup_shader_code = (
    shader: string,
    p: bigint,
    num_words: number,
    word_size: number,
    n0: bigint,
    cost: number,
) => {
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        shader,
        {
            word_size,
            num_words,
            n0,
            cost,
            p_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
        },
        {
            bigint_struct,
            bigint_funcs,
            montgomery_product_funcs,
        },
    )
    //console.log(shaderCode)
    return shaderCode
}

const expensive_computation = (
    a: ExtPointType,
    b: ExtPointType,
    cost: number,
    fieldMath: FieldMath,
): ExtPointType => {
    let c = add_points_any_a(a, b, fieldMath)
    for (let i = 1; i < cost; i ++) {
        c = add_points_any_a(c, a, fieldMath)
    }
    //let c = add_points_a_minus_one(a, b, fieldMath)
    //for (let i = 1; i < cost; i ++) {
        //c = add_points_a_minus_one(c, a, fieldMath)
    //}

    return c
}

// Ignore the input points and scalars. Generate two random Extended
// Twisted Edwards points, perform many additions, and print the time taken.
export const add_points_any_a_benchmark = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const cost = 10240 * 2
    const fieldMath = new FieldMath();
    fieldMath.aleoD = BigInt(-1)
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

    // pt is not the generator of the group. it's just a valid curve point.
    const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246');
    const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166');
    const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023');
    const z = BigInt('1');
    const pt = fieldMath.createPoint(x, y, t, z)

    // Generate two random points by multiplying by random field elements
    const a = pt.multiply(genRandomFieldElement(p))
    const b = pt.multiply(genRandomFieldElement(p))

    // Compute in CPU
    // Start timer
    const start_cpu = Date.now()
    const expected_cpu = expensive_computation(a, b, cost, fieldMath)
    const expected_cpu_affine = expected_cpu.toAffine()
    // End Timer
    const elapsed_cpu = Date.now() - start_cpu
    console.log(`CPU took ${elapsed_cpu}ms`)

    const num_x_workgroups = 1;
    const word_size = 13
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    const r = params.r
    const rinv = params.rinv

    // Convert to Mont form
    const points_with_mont_coords: BigIntPoint[] = []
    for (const pt of [a, b]) {
        points_with_mont_coords.push(
            {
                x: fieldMath.Fp.mul(pt.ex, r),
                y: fieldMath.Fp.mul(pt.ey, r),
                t: fieldMath.Fp.mul(pt.et, r),
                z: fieldMath.Fp.mul(pt.ez, r),
            }
        )
    }

    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)

    const device = await get_device()
    const commandEncoder = device.createCommandEncoder();

    //add_points_a_minus_one_shader,
    //add_points_any_a_shader,
 
    const shaderCode = setup_shader_code(add_points_any_a_shader, p, num_words, word_size, n0, cost)
    const shaderModule = device.createShaderModule({ code: shaderCode })

    const start_gpu = Date.now()

    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ]
    })

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [ points_storage_buffer ],
    )

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // Initiate compute pass
    const passEncoder = commandEncoder.beginComputePass();

    // Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(num_x_workgroups)

    // End the render pass
    passEncoder.end();

    // Create buffer to read result
    const stagingBuffer = device.createBuffer({
        size: points_storage_buffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(
        points_storage_buffer, // source
        0, // sourceOffset
        stagingBuffer, // destination
        0, // destinationOffset
        points_storage_buffer.size,
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        points_storage_buffer.size,
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, points_storage_buffer.size)
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();
    const elapsed_gpu = Date.now() - start_gpu

    console.log(`GPU took ${elapsed_gpu}ms`)

    const data_as_uint8s = new Uint8Array(data)

    const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
        return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z)
    }

    const output_points = u8s_to_points(data_as_uint8s, num_words, word_size)

    // convert output_points out of Montgomery coords
    const output_points_non_mont: ExtPointType[] = []
    for (const pt of output_points) {
        const non = {
            x: fieldMath.Fp.mul(pt.x, rinv),
            y: fieldMath.Fp.mul(pt.y, rinv),
            t: fieldMath.Fp.mul(pt.t, rinv),
            z: fieldMath.Fp.mul(pt.z, rinv),
        }
        output_points_non_mont.push(bigIntPointToExtPointType(non))
    }
    // convert output_points_non_mont into affine
    const output_points_non_mont_and_affine = output_points_non_mont.map((x) => x.toAffine())
    console.log('result from gpu, in affine:', output_points_non_mont_and_affine[0])

    console.log(expected_cpu.toAffine())

    assert(output_points_non_mont_and_affine[0].x === expected_cpu_affine.x)
    assert(output_points_non_mont_and_affine[0].y === expected_cpu_affine.y)

    return { x: BigInt(1), y: BigInt(0) }
}
