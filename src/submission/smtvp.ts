// TODO:
//  1. Convert point coordinates to Montgomery form, write test shader for this
import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix, CSRSparseMatrix } from './matrices/matrices'; 
import store_point_at_infinity_shader from '../submission/wgsl/store_point_at_infinity.template.wgsl'
import smtvp_shader from '../submission/wgsl/smtvp.template.wgsl'
import bigint_struct from '../submission/wgsl/structs/bigint.template.wgsl'
import { u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu } from './utils'
import assert from 'assert'

export async function get_device() {
    const gpuErrMsg = "Please use a browser that has WebGPU enabled.";
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
    });
    if (!adapter) {
        console.log(gpuErrMsg)
        throw Error('Couldn\'t request WebGPU adapter.')
    }

    const device = await adapter.requestDevice()
    return device
}

export async function gen_csr_sparse_matrices(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    lambda: number, // λ-bit scalars
    s: number, // s-bit window size 
    threads: number, // Thread count
): Promise<any> {  
    // Number of rows and columns (ie. row-space)
    const num_rows = threads
    const num_columns = Math.pow(2, s) - 1

    // Instantiate 'FieldMath' object
    const fieldMath = new FieldMath();
    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const csr_sparse_matrix_array: CSRSparseMatrix[] = []

    for (let i = 0; i < num_rows; i++) {
        // Instantiate empty ELL sparse matrix format
        const data = new Array(num_rows);
        for (let i = 0; i < num_rows; i++) {
            data[i] = new Array(num_columns).fill(ZERO_POINT);
        }

        const col_idx = new Array(num_rows);
        for (let i = 0; i < num_rows; i++) {
            col_idx[i] = new Array(num_columns).fill(0);
        }

        const row_length = Array(num_rows).fill(0);

        // Perform scalar decomposition
        const scalars_decomposed: number[][] = []
        for (let j =  Math.ceil(lambda / s); j > 0; j--) {
            const chunk: number[] = [];
            for (let i = 0; i < scalars.length; i++) {
                const mask = (BigInt(1) << BigInt(s)) - BigInt(1)  
                const limb = (scalars[i] >> BigInt(((j - 1) * s))) & mask // Right shift and extract lower 32-bits 
                chunk.push(Number(limb))
            }
            scalars_decomposed.push(chunk);
        }

        // Divide EC points into t parts
        for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
            for (let j = 0; j < num_columns; j++) {
                const point_i = thread_idx + j * threads
                data[thread_idx][j] = baseAffinePoints[point_i]
                col_idx[thread_idx][j] = scalars_decomposed[i][point_i]
                row_length[thread_idx] += 1
            }
        }

        // Transform ELL sparse matrix to CSR sparse matrix
        const ell_sparse_matrix = new ELLSparseMatrix(data, col_idx, row_length)
        const csr_sparse_matrix = await new CSRSparseMatrix([], [], []).ell_to_csr_sparse_matrix(ell_sparse_matrix)

        csr_sparse_matrix_array.push(csr_sparse_matrix)
    }

    return csr_sparse_matrix_array
}

export const smtvp = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    // Decompose the scalars into windows. In the actual implementation, this
    // should be done by a shader.
    const word_size = 13
    //const params = compute_misc_params(word_size)
    const num_words = 20

    // λ-bit scalars. 13 * 20 = 260
    const lambda = word_size * num_words

    // s-bit window size 
    const s = word_size

    // Thread count
    const threads = 16

    const csr_sparse_matrices = await gen_csr_sparse_matrices(
        baseAffinePoints,
        scalars,
        lambda,
        s,
        threads
    )

    const device = await get_device()
    for (let i = 0; i < csr_sparse_matrices.length; i ++) {
        const max_col_idx = Math.max(...csr_sparse_matrices[i].col_idx, num_words)

        // Shader 1: store the point at infinity in the output buffer
        const output_storage_buffer = await store_point_at_infinity_shader_gpu(device, max_col_idx, num_words)

        // Shader 2: perform SMTVP
        await smtvp_gpu(device, csr_sparse_matrices[i], num_words, word_size, output_storage_buffer)
        break
    }

    return { x: BigInt(1), y: BigInt(0) }
}

export async function store_point_at_infinity_shader_gpu(
    device: GPUDevice,
    max_col_idx: number,
    num_words: number,
) {
    const num_x_workgroups = 256;

    const output_buffer_length = max_col_idx * num_words * 4 * 4

    const shaderCode = mustache.render(store_point_at_infinity_shader, { num_words })

    //console.log(shaderCode)

    // 2: Create a shader module from the shader template literal
    const shaderModule = device.createShaderModule({
        code: shaderCode
    })

    const output_storage_buffer = device.createBuffer({
        size: output_buffer_length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                },
            },
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: output_storage_buffer,
                }
            },
        ]
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = device.createCommandEncoder();

    const start = Date.now()
    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(num_x_workgroups)

    // End the render pass
    passEncoder.end();

    const stagingBuffer = device.createBuffer({
        size: output_buffer_length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    commandEncoder.copyBufferToBuffer(
        output_storage_buffer, // source
        0, // sourceOffset
        stagingBuffer, // destination
        0, // destinationOffset
        output_buffer_length // size
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        output_buffer_length,
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, output_buffer_length)
    const data = copyArrayBuffer.slice(0);

    const fieldMath = new FieldMath();
    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const data_as_uint8s = new Uint8Array(data)

    stagingBuffer.unmap();

    const elapsed = Date.now() - start
    console.log(`GPU took ${elapsed}ms`)

    // TODO: skip this check in production
    for (const point of u8s_to_points(data_as_uint8s, num_words)) {
        assert(point.x === ZERO_POINT.ex)
        assert(point.y === ZERO_POINT.ey)
        assert(point.t === ZERO_POINT.et)
        assert(point.z === ZERO_POINT.ez)
    }
    return output_storage_buffer
}

export async function smtvp_gpu(
    device: GPUDevice,
    csr_sm: CSRSparseMatrix,
    num_words: number,
    word_size: number,
    previous_output_buffer: GPUBuffer,
) {
    const num_x_workgroups = 1;

    const max_col_idx = Math.max(...csr_sm.col_idx)
    const output_buffer_length = max_col_idx * num_words * 4 * 4

    const col_idx_bytes = numbers_to_u8s_for_gpu(csr_sm.col_idx)
    const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sm.row_ptr)
    const points_bytes = points_to_u8s_for_gpu(csr_sm.data, num_words, word_size)

    const num_rows = csr_sm.row_ptr.length - 1

    const shaderCode = mustache.render(
        smtvp_shader,
        {
            num_words,
            num_rows,
        },
        {
            bigint_struct,
        },
    )

    //console.log(shaderCode)

    // 2: Create a shader module from the shader template literal
    const shaderModule = device.createShaderModule({
        code: shaderCode
    })

    const output_storage_buffer = device.createBuffer({
        size: output_buffer_length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    const col_idx_storage_buffer = device.createBuffer({
        size: col_idx_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(col_idx_storage_buffer, 0, col_idx_bytes);

    const row_ptr_storage_buffer = device.createBuffer({
        size: row_ptr_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(row_ptr_storage_buffer, 0, row_ptr_bytes);

    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);

    const stagingBuffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                },
            },
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: output_storage_buffer,
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: col_idx_storage_buffer,
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: row_ptr_storage_buffer,
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: points_storage_buffer,
                }
            },
        ]
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = device.createCommandEncoder();

    const start = Date.now()

    // Copy the previous shader's output buffer to the output buffer of this
    // shader. The values should be the point at infinity.
    commandEncoder.copyBufferToBuffer(
        previous_output_buffer, // source
        0, // sourceOffset
        output_storage_buffer, // destination
        0, // destinationOffset
        previous_output_buffer.size // size
    );


    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(num_x_workgroups)

    // End the render pass
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
        output_storage_buffer, // source
        0, // sourceOffset
        stagingBuffer, // destination
        0, // destinationOffset
        output_buffer_length // size
    );
    // 8: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        points_bytes.length
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, points_bytes.length)
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();

    const elapsed = Date.now() - start

    console.log(`GPU took ${elapsed}ms`)

    const data_as_uint8s = new Uint8Array(data)

    console.log(new Uint32Array(points_bytes))
    console.log(u8s_to_points(data_as_uint8s, num_words)[0])
}
