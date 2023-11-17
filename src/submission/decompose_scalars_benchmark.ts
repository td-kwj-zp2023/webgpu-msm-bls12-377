import assert from 'assert'
import mustache from 'mustache'
import * as wasm from 'decompose-scalars'
import {toBufferBE, toBufferLE} from 'bigint-buffer'
import { BigIntPoint } from "../reference/types"
import {
    to_words_le,
    u8s_to_numbers,
    decompose_scalars,
    compute_misc_params,
} from './utils'
import { get_device, create_bind_group } from './gpu'
import decompose_scalars_shader from './wgsl/decompose_scalars.template.wgsl'

export const decompose_scalars_ts_benchmark = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const num_words = 20
    const word_size = 13

    /*
    console.log('Typescript benchmarks:')
    //for (let word_size = 13; word_size < 14; word_size ++) {
    for (let word_size = 8; word_size < 20; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start = Date.now()
        decompose_scalars(scalars, num_words, word_size)
        const elapsed = Date.now() - start
        console.log(`decompose_scalars() with ${word_size}-bit windows took ${elapsed}ms`)
    }
    console.log()

    console.log('WASM benchmarks:')
    //for (let word_size = 13; word_size < 14; word_size ++) {
    for (let word_size = 8; word_size < 20; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start_wasm = Date.now()
        wasm.decompose_scalars(scalars, num_words, word_size).get_result()
        const elapsed_wasm = Date.now() - start_wasm
        console.log(`WASM with ${word_size}-bit windows took ${elapsed_wasm}ms`)
    }

    const ts_r = decompose_scalars(scalars, num_words, word_size).flat()
    const wasm_r = wasm.decompose_scalars(scalars, num_words, word_size).get_result()
    assert(ts_r.toString() === wasm_r.toString())
    console.log('ok')

    //debugger
    */

    console.log('GPU benchmarks:')
    const ts_r = decompose_scalars(scalars, num_words, word_size).flat()
    console.log(ts_r)
    await decompose_scalars_gpu(scalars, 20, 13)

    return { x: BigInt(0), y: BigInt(0) }
}

const decompose_scalars_gpu = async (
    scalars: bigint[],
    num_words: number,
    word_size: number,
) => {
    //console.log(scalars)
    // Convert scalars to bytes
    const scalar_bytes = new Uint8Array(scalars.length * 32)
    for (let i = 0; i < scalars.length; i ++) {
        const scalar = scalars[i]
        const buf = toBufferBE(scalar, 32)
        scalar_bytes.set(buf, i * 32)
    }

    const device = await get_device()
    const num_x_workgroups = 2

    const scalars_storage_buffer = device.createBuffer({
        size: scalar_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(scalars_storage_buffer, 0, scalar_bytes)

    // Output buffers
    const result_storage_buffer = device.createBuffer({
        size: num_words * scalars.length * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    })

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [scalars_storage_buffer, result_storage_buffer]
    )

    const shaderCode = mustache.render(
        decompose_scalars_shader,
        {
            num_words,
            word_size,
        },
        {
        }
    )

    const shaderModule = device.createShaderModule({ code: shaderCode })

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    })

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups)
    passEncoder.end()

    const result_staging_buffer = device.createBuffer({
        size: result_storage_buffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    commandEncoder.copyBufferToBuffer(
        result_storage_buffer,
        0,
        result_staging_buffer,
        0,
        result_storage_buffer.size
    )
    device.queue.submit([commandEncoder.finish()]);

    // map staging buffers to read results back to JS
    await result_staging_buffer.mapAsync(
        GPUMapMode.READ,
        0,
        result_storage_buffer.size
    )

    const result_data = result_staging_buffer.getMappedRange(0, result_staging_buffer.size).slice(0)
    result_staging_buffer.unmap()

    const scalar_chunks = u8s_to_numbers(new Uint8Array(result_data))
    console.log(scalar_chunks)
}
