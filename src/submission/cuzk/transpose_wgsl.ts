import { BigIntPoint } from "../../reference/types";
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { CSRSparseMatrix, ELLSparseMatrix, fieldMath } from '../matrices/matrices';
import transpose_shader from '../wgsl/cuzk/transpose_segment_1.template.wgsl'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_functions from '../wgsl/bigint/bigint.template.wgsl'
import curve_functions from '../wgsl/curve/ec.template.wgsl'
import curve_parameters from '../wgsl/curve/parameters.template.wgsl'
import field_functions from '../wgsl/field/field.template.wgsl'
import montgomery_product_functions from '../wgsl/montgomery/mont_pro_product.template.wgsl'
import { u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu, compute_misc_params, to_words_le } from '../utils'
import assert from 'assert'

export async function transpose(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> {
    // Scalar finite field
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

    // s-bit window size 
    const word_size = 13

    // Number of limbs (ie. windows)
    const num_words = 20

    // λ-bit scalars (13 * 20 = 260)
    const lambda = 260

    // Thread count
    const threads = 16

    // Misc parameters
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0    

    const edwards_limbs = to_words_le(fieldMath.Fp.mul(BigInt(3021), params.r), 20, 13)
    console.log("! edwards_limbs: ", edwards_limbs)

    // Request GPU device
    const device = await get_device()

    // Generate CSR sparse matrices
    const csr_sparse_matrices = await gen_csr_sparse_matrices(
        baseAffinePoints,
        scalars,
        lambda,
        word_size,
        threads
    )

    // WGSL Shader invocations
    for (let i = 0; i < csr_sparse_matrices.length; i ++) {
        // Determine maximum column index
        let max_col_idx = 0
        for (const j of csr_sparse_matrices[i].col_idx) {
            if (j > max_col_idx) {
                max_col_idx = j
            }
        }

        // Perform Sparse-Matrix Tranpose and SMVP
        await transpose_gpu(device, csr_sparse_matrices[i], num_words, word_size, p, n0, params.r, params.rinv, max_col_idx)
        break
    }
    return { x: BigInt(1), y: BigInt(0) }
}

export async function gen_csr_sparse_matrices(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    lambda: number, 
    s: number, 
    threads: number, 
): Promise<any> {  
    // Number of rows and columns (ie. row-space)
    const num_rows = threads
    const num_columns = Math.pow(2, s) - 1
  
    // Intantiate empty array of sparse matrices
    const csr_sparse_matrix_array: CSRSparseMatrix[] = []
    
    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;
    console.log("zero point is: ", ZERO_POINT)
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

export async function transpose_gpu(
    device: GPUDevice,
    csr_sm: CSRSparseMatrix,
    num_words: number,
    word_size: number,
    p: bigint,
    n0: bigint,
    r: bigint,
    rinv: bigint,
    max_col_idx: number
) {
    // Convert BigIntPoint to montgomery form
    const points_with_mont_coords: BigIntPoint[] = []
    for (const pt of csr_sm.data) {
        points_with_mont_coords.push(
            {
                x: fieldMath.Fp.mul(pt.x, r),
                y: fieldMath.Fp.mul(pt.y, r),
                t: fieldMath.Fp.mul(pt.t, r),
                z: fieldMath.Fp.mul(pt.z, r),
            }
        )
    }

    // Define number of workgroups
    const num_x_workgroups = 1;

    const col_idx_bytes = numbers_to_u8s_for_gpu(csr_sm.col_idx)
    const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sm.row_ptr)
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const num_rows = csr_sm.row_ptr.length - 1

    // 1: Create a shader module with templating engine
    const shaderCode = mustache.render(
        transpose_shader,
        {
            num_words,
            word_size,
            num_rows,
            max_col_idx,
            n0,
            two_pow_word_size: 2 ** word_size,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
        },
        {
            structs,
            bigint_functions,
            montgomery_product_functions,
            curve_functions,
            curve_parameters,
            field_functions,
        },
    )
    const shaderModule = device.createShaderModule({
        code: shaderCode
    })

    // 2: Create buffered memory accessible by the GPU memory space
    // const output_buffer_length = max_col_idx * num_words * 4 * 4
    const output_buffer_length = 640

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

    // 3: Define bind group layouts and bind groups 
    // Bind Group Layout defines the input/output interface expected by the shader 
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ]
    });

    // Bind Group represents the actual input/output data for the shader, and associate with GPU buffers 
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: output_storage_buffer } },
            { binding: 1, resource: { buffer: col_idx_storage_buffer } },
            { binding: 2, resource: { buffer: row_ptr_storage_buffer } },
            { binding: 3, resource: { buffer: points_storage_buffer } },
        ]
    });

    // 4: Setup Compute Pipeline 
    // Creates pipeline with bind group layout and compute stage as arguments
    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute : {
            module: shaderModule,   
            entryPoint: "main"     
        }
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    // Returns a Javascript object that encodes a batch of "buffered" GPU commands 
    const commandEncoder = device.createCommandEncoder();

    // Start timer
    const start = Date.now()

    // 6: Encode pipeline commands 
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);

    // Set bind group at index 0 (corresponding with group(0) in WGSL) 
    passEncoder.setBindGroup(0, bindGroup);

    // Set the number of workgroups dispatched for the execution of a kernel function 
    passEncoder.dispatchWorkgroups(num_x_workgroups)

    // End the render pass
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
        output_storage_buffer,      // source
        0,                          // sourceOffset
        stagingBuffer,              // destination
        0,                          // destinationOffset
        output_buffer_length / 2        // buffer size
    );

    // 8: Finish encoding commands and submit to GPU device command queue
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // 9: Map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        points_bytes.length,
    );

    // Returns an ArrayBuffer with the contents of the GPUBuffer in the given mapped range.
    const JSArrayBuffer = stagingBuffer.getMappedRange(0, output_buffer_length)
    const data = JSArrayBuffer.slice(0);
    stagingBuffer.unmap();

    // End Timer
    const elapsed = Date.now() - start
    console.log(`GPU took ${elapsed}ms`)

    // Transform results 
    const data_as_uint8s = new Uint8Array(data)
    const output_points = u8s_to_points(data_as_uint8s, num_words, word_size)
    const gpu_point = fieldMath.createPoint(output_points[0].x, output_points[0].y, output_points[0].t, output_points[0].z)
    const gpu_point_affine = gpu_point.toAffine()

    // Compare agaisnt expected CPU results 
    const expected_1 = add_points(points_with_mont_coords[0], points_with_mont_coords[1], p, rinv, r)
    const expected_1_affine = expected_1.toAffine()

    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;
    const test_point = fieldMath.createPoint(csr_sm.data[0].x, csr_sm.data[0].y, csr_sm.data[0].t, csr_sm.data[0].z)
    const result = test_point.add(test_point)
    const expected_affine_2 = result.toAffine()

    // Assertion checks
    assert(expected_1_affine.x === gpu_point_affine.x)
    assert(expected_1_affine.y === gpu_point_affine.y)
    assert(expected_affine_2.x === gpu_point_affine.x)
    assert(expected_affine_2.y === gpu_point_affine.y)

    // Print results
    console.log('GPU results:', gpu_point_affine)
    console.log('CPU results 1:', expected_1_affine)
    console.log('CPU results 1:', expected_affine_2)
}

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

export const add_points = (
    p1: BigIntPoint,
    p2: BigIntPoint,
    p: bigint,
    rinv: bigint,
    r: bigint
): ExtPointType => {
    const montgomery_product = (
        a: bigint,
        b: bigint,
    ): bigint => {
        const fp = fieldMath.Fp
        const ab = fp.mul(a, b)
        const abr = fp.mul(ab, rinv)
        return abr
    }

    // Typedef functions
    const fr_add = fieldMath.Fp.add
    const fr_sub = fieldMath.Fp.sub

    const a = montgomery_product(p1.x, p2.x)
    const b = montgomery_product(p1.y, p2.y)
    const t2 = montgomery_product(p1.t, p2.t)
    const EDWARDS_D = BigInt(3021) * r
    const c = montgomery_product(EDWARDS_D, t2)
    const d = montgomery_product(p1.z, p2.z)
    const p1_added = fr_add(p1.x, p1.y)
    const p2_added = fr_add(p2.x, p2.y)
    let e = montgomery_product(p1_added, p2_added)
    e = fr_sub(e, a)
    e = fr_sub(e, b);
    const f = fr_sub(d, c)
    const g = fr_add(d, c)
    const a_neg = fr_sub(p, a)
    const h = fr_sub(b, a_neg)
    const added_x = montgomery_product(e, f);
    const added_y = montgomery_product(g, h);
    const added_t = montgomery_product(e, h);
    const added_z = montgomery_product(f, g);

    return fieldMath.createPoint(added_x, added_y, added_t, added_z)
}
