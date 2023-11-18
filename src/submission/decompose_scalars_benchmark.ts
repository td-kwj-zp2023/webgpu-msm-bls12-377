import assert from 'assert'
import mustache from 'mustache'
import * as wasm from 'decompose-scalars'
import { BigIntPoint } from "../reference/types"
import {
    to_words_le,
    u8s_to_numbers,
    decompose_scalars,
    compute_misc_params,
    bigints_to_u8_for_gpu,
} from './utils'
import { get_device, create_bind_group } from './gpu'
import decompose_scalars_shader from './wgsl/decompose_scalars.template.wgsl'

export const decompose_scalars_ts_benchmark = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    console.log('Decomposing', scalars.length, 'scalars')

    console.log('Typescript benchmarks:')
    for (let word_size = 8; word_size < 17; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start = Date.now()
        decompose_scalars(scalars, num_words, word_size)
        const elapsed = Date.now() - start
        console.log(`decompose_scalars() with ${word_size}-bit windows took ${elapsed}ms`)
    }
    console.log()

    console.log('WASM benchmarks:')
    for (let word_size = 8; word_size < 17; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words

        const start_wasm = Date.now()
        wasm.decompose_scalars(scalars, num_words, word_size).get_result()
        const elapsed_wasm = Date.now() - start_wasm
        console.log(`WASM with ${word_size}-bit windows took ${elapsed_wasm}ms`)
    }
    console.log()

    const num_words = 20
    const word_size = 13
    const ts_r = decompose_scalars(scalars, num_words, word_size).flat()
    const wasm_r = wasm.decompose_scalars(scalars, num_words, word_size).get_result()
    assert(ts_r.toString() === wasm_r.toString())
    //console.log('ok')

    console.log('GPU benchmarks:')
   for (let word_size = 8; word_size < 17; word_size ++) {
        const params = compute_misc_params(p, word_size)
        const num_words = params.num_words
        await decompose_scalars_gpu(scalars, num_words, word_size)
    }

    return { x: BigInt(0), y: BigInt(0) }
}

const extract_word_from_bytes_le = (
    input: number[],
    word_idx: number,
    word_size: number,
): number => {
    /*
     * If word_size == 13:
     * ---- ---- ---- ---- .... .... .... ....
     * ++++ ++xx xxxx xxxx xxx+ ++++ ++++ ++++
     *
     * ---- ---- ---- ---- .... .... .... ....
     * ++++ ++++ ++++ xxxx xxxx xxxx x+++ ++++
     *
     * ---- ---- ---- ---- .... .... .... ....
     * xxxx x+++ ++++ ++++ ++xx xxxx xxxx xxx+
     *
     * 15, 13, 15, 0
     * 14, 10, 15, 13
     * 13,  7, 14, 10
     * 12,  4, 13,  7
     * 11,  1, 12,  4
     * 11, 14, 11,  1
     */
    const start_byte_idx = 15 - Math.floor((word_idx * word_size + word_size) / 16)
    const end_byte_idx = 15 - Math.floor((word_idx * word_size) / 16)

    const start_byte_offset = (word_idx * word_size + word_size) % 16
    const end_byte_offset = (word_idx * word_size) % 16

    const mask = 2 ** start_byte_offset - 1
    let word = 0

    if (start_byte_idx === end_byte_idx) {
        word = (input[start_byte_idx] & mask) >> end_byte_offset
    } else {
        word = (input[start_byte_idx] & mask) << (16 - end_byte_offset) 
        word += input[end_byte_idx] >> end_byte_offset
    }

    return word
}

const bigint_to_words_32 = (val: bigint) => {
    let hex_str = val.toString(16)
    while (hex_str.length < 64) {
        hex_str = '0' + hex_str
    }
    const double_bytes: number[] = []
    for (let i = 0; i < hex_str.length; i += 4) {
        const db = Number('0x' + hex_str.slice(i, i + 4))
        double_bytes.push(db)
    }
    return double_bytes
}

const to_words_le_modified = (
    val: bigint,
    num_words: number,
    word_size: number,
): Uint16Array => {
    const double_bytes = bigint_to_words_32(val)

    const words = Array(num_words).fill(0)
    for (let i = 0; i < num_words - 1; i ++) {
        const w = extract_word_from_bytes_le(double_bytes, i, word_size)
        words[i] = w
    }
    
    words[num_words - 1] = double_bytes[0] >> (((num_words * word_size - 256) + 16) - word_size)
    return new Uint16Array(words)
}

const decompose_scalars_gpu = async (
    scalars: bigint[],
    num_words: number,
    word_size: number,
) => {
    // Convert scalars to bytes
    const scalar_bytes = bigints_to_u8_for_gpu(scalars, 16, 16)

    // Calculate expected decomposed scalars
    const expected: number[][] = []
    const expected2: number[][] = []
    for (let i = 0; i < scalars.length; i ++) {
        expected.push(Array.from(to_words_le_modified(scalars[i], num_words, word_size)))
        expected2.push(Array.from(to_words_le(scalars[i], num_words, word_size)))
    }
    assert(expected.toString() === expected2.toString())

    const device = await get_device()
    const workgroup_size = 64
    const num_x_workgroups = 256
    const num_y_workgroups = scalars.length / workgroup_size / num_x_workgroups

    const scalars_storage_buffer = device.createBuffer({
        size: scalar_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(scalars_storage_buffer, 0, scalar_bytes)

    // Output buffers
    const result_storage_buffer = device.createBuffer({
        size: num_words * scalars.length * 4,
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
            workgroup_size,
            num_y_workgroups,
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

    const start = Date.now();

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, 1)
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

    const elapsed = Date.now() - start
    console.log(`GPU with ${word_size}-bit windows took ${elapsed}ms`)

    const result_data = result_staging_buffer.getMappedRange(0, result_staging_buffer.size).slice(0)
    result_staging_buffer.unmap()

    const scalar_chunks = u8s_to_numbers(new Uint8Array(result_data))
    //console.log('from gpu:', scalar_chunks)
    for (let i = 0; i < expected.length; i ++) {
        for (let j = 0; j < num_words; j ++) {
            const e = expected[i][j]
            const s = scalar_chunks[i * num_words + j]
            if (e !== s) {
                console.log(i, j)
                debugger
            }
            assert(e === s)
        }
    }
}
