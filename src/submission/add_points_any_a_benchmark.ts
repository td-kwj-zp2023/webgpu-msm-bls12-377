import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../reference/utils/FieldMath";
import { genRandomFieldElement } from './utils'
import { compute_misc_params, u8s_to_points, points_to_u8s_for_gpu, gen_p_limbs } from './utils'
import add_points_any_a_shader from '../submission/wgsl/add_points_any_a.template.wgsl'
import bigint_struct from '../submission/wgsl/structs/bigint.template.wgsl'
import bigint_funcs from '../submission/wgsl/bigint.template.wgsl'
import montgomery_product_funcs from '../submission/wgsl/montgomery_product.template.wgsl'
import { get_device, create_bind_group } from './gpu'

const setup_shader_code = (
    p: bigint,
    num_words: number,
    word_size: number,
    n0: bigint,
    cost: number,
) => {
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        add_points_any_a_shader,
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
): ExtPointType => {
    let c = a.add(b)
    for (let i = 1; i < cost; i ++) {
        c = c.add(a)
    }
    return c
}

// Ignore the input points and scalars. Generate two random Extended
// Twisted Edwards points, perform many additions, and print the time taken.
export const add_points_any_a_benchmark = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const cost = 10240
    const fieldMath = new FieldMath();
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
    const expected_cpu = expensive_computation(a, b, cost)
    // End Timer
    const elapsed_cpu = Date.now() - start_cpu
    console.log(`CPU took ${elapsed_cpu}ms`)

    const num_x_workgroups = 1;
    const word_size = 13
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    const r = params.r

    // TODO: convert to Mont form
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
    const shaderCode = setup_shader_code(p, num_words, word_size, n0, cost)
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

    console.log(expected_cpu)
    console.log(output_points)

    return { x: BigInt(1), y: BigInt(0) }
}
