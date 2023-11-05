import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import {
    gen_p_limbs,
    u8s_to_points,
    u8s_to_bigints,
    compute_misc_params,
    points_to_u8s_for_gpu,
    bigints_to_u8_for_gpu,
} from './utils'
import { get_device, create_bind_group } from './gpu'
import structs from './wgsl/struct/structs.template.wgsl'
import bigint_funcs from './wgsl/bigint/bigint.template.wgsl'
import field_funcs from './wgsl/field/field.template.wgsl'
import convert_inputs_shader from './wgsl/convert_inputs.template.wgsl'
import montgomery_product_funcs from './wgsl/montgomery/mont_pro_product.template.wgsl'

export const convert_inputs_into_mont_benchmark = async(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const num_x_workgroups = 256
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const word_size = 13
    const params = compute_misc_params(p, word_size)
    const num_words = params.num_words
    const r = params.r
    const n0 = params.n0

    const start = Date.now()

    const converted_scalar = []
    const converted_x = []
    const converted_y = []
    const converted_t = []
    const converted_z = []

    for (const scalar of scalars) {
        const a = (scalar * r) % p
        converted_scalar.push(a)
    }

    for (const pt of baseAffinePoints) {
        const xr = (pt.x * r) % p
        const yr = (pt.y * r) % p
        const tr = (xr * pt.y) % p
        converted_x.push(xr)
        converted_y.push(yr)
        converted_t.push(tr)
        converted_z.push(r)
    }

    const elapsed = Date.now() - start
    console.log(`CPU (serial) took ${elapsed}ms to convert ${scalars.length} scalars and points`)

    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        convert_inputs_shader,
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
            montgomery_product_funcs,
        },
    )

    //console.log(shaderCode)

    const points_bytes = points_to_u8s_for_gpu(baseAffinePoints, num_words, word_size)
    const scalars_bytes = bigints_to_u8_for_gpu(scalars, num_words, word_size)
    
    const device = await get_device()
    const shaderModule = device.createShaderModule({ code: shaderCode })

    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const scalars_storage_buffer = device.createBuffer({
        size: scalars_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);
    device.queue.writeBuffer(scalars_storage_buffer, 0, scalars_bytes);

    const points_staging_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })
    const scalars_staging_buffer = device.createBuffer({
        size: scalars_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ]
    })

    const bindGroup = create_bind_group(device, bindGroupLayout, [points_storage_buffer, scalars_storage_buffer])

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    })

    const start_gpu = Date.now()
    // Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = device.createCommandEncoder();

    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(num_x_workgroups)

    // End the render pass
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
        points_storage_buffer,
        0, // Source offset
        points_staging_buffer,
        0, // Destination offset
        points_bytes.length
    )

    commandEncoder.copyBufferToBuffer(
        scalars_storage_buffer,
        0, // Source offset
        scalars_staging_buffer,
        0, // Destination offset
        scalars_bytes.length
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    const elapsed_gpu = Date.now() - start
    console.log(`GPU took ${elapsed_gpu}ms (including data transfer time)`)

    await points_staging_buffer.mapAsync(GPUMapMode.READ, 0, points_bytes.length)
    await scalars_staging_buffer.mapAsync(GPUMapMode.READ, 0, scalars_bytes.length)

    const points_array_buffer = points_staging_buffer.getMappedRange(0, points_bytes.length)
    const points_data = points_array_buffer.slice(0)
    points_staging_buffer.unmap()

    const scalars_array_buffer = scalars_staging_buffer.getMappedRange(0, scalars_bytes.length)
    const scalars_data = scalars_array_buffer.slice(0)
    scalars_staging_buffer.unmap()

    const points_from_gpu = u8s_to_points(new Uint8Array(points_data), num_words, word_size)
    const scalars_from_gpu = u8s_to_bigints(new Uint8Array(scalars_data), num_words, word_size)

    return { x: BigInt(0), y: BigInt(0) }
}
