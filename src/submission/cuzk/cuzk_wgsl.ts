import { BigIntPoint } from "../../reference/types";
import { bigints_to_u8_for_gpu } from '../../submission/utils'
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { CSRSparseMatrix, ELLSparseMatrix, fieldMath } from '../matrices/matrices';
import store_point_at_infinity_shader from '../wgsl/store_point_at_infinity.template.wgsl'
import bigint_struct from '../submission/wgsl/bigint.template.wgsl'
// import shader from '../wgsl/mont_pro_optimised.template.wgsl'
import { u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu } from '../utils'
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

    // Returns a promise that asynchronously resolves with a GPU device
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
  
    // Intantiate empty array of sparse matrices
    const csr_sparse_matrix_array: CSRSparseMatrix[] = []
    
    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;
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
      const scalars_decomposed: bigint[][] = []
      for (let j =  Math.ceil(lambda / s); j > 0; j--) {
        const chunk: bigint[] = [];
        for (let i = 0; i < scalars.length; i++) {
          const mask = (BigInt(1) << BigInt(s)) - BigInt(1)  
          const limb = (scalars[i] >> BigInt(((j - 1) * s))) & mask // Right shift and extract lower 32-bits 
          chunk.push(limb)
        }
        scalars_decomposed.push(chunk);
      }
      
      // Divide EC points into t parts
      for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
        const z = 0
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

export async function execute_cuzk_wgsl(
    inputSize: number,
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<any> {    
    // s-bit window size 
    const s = 13

    // Number of limbs (ie. windows)
    const num_limbs = 20

    // λ-bit scalars (13 * 20 = 260)
    const lambda = 260

    // Thread count
    const threads = 16

    // 1. Request device
    const device = await get_device()

    // 2. Generate CSR sparse matrices
    const csr_sparse_matrices = await gen_csr_sparse_matrices(
        baseAffinePoints,
        scalars,
        lambda,
        s,
        threads
    )

    // 3. Shader invocations
    for (let i = 0; i < csr_sparse_matrices.length; i ++) {
        // Determine maximum column index
        // const max_col_idx = Math.max(Number(csr_sparse_matrices[0].col_idx), num_limbs)
        let max_col_idx = 0
        for (const j of csr_sparse_matrices[i].col_idx) {
            if (j > max_col_idx) {
                max_col_idx = j
            }
        }

        console.log("Number(csr_sparse_matrices[0].col_idx) is: ", max_col_idx)

        // Shader 1: store the point at infinity in the output buffer
        const output_storage_buffer = await store_point_at_infinity_shader_gpu(device, Number(max_col_idx), num_limbs)

        // Shader 2: perform Sparse-Matrix Tranpose and SMVP
        // await smtvp_gpu(device, csr_sparse_matrices[i], num_limbs, s, output_storage_buffer)
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

    console.log(shaderCode)

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

    console.log("passed assertion checks!")
    return output_storage_buffer
}

