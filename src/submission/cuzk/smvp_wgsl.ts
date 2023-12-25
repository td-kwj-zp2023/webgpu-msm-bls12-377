import { BigIntPoint } from "../../reference/types";
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { CSRSparseMatrix, ELLSparseMatrix, fieldMath } from '../matrices/matrices';
import smvp_shader from '../wgsl/cuzk/smvp.template.wgsl'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_functions from '../wgsl/bigint/bigint.template.wgsl'
import field_functions from '../wgsl/field/field.template.wgsl'
import curve_functions from '../wgsl/curve/ec.template.wgsl'
import curve_parameters from '../wgsl/curve/parameters.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery/mont_pro_product.template.wgsl'
import { u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu, compute_misc_params, to_words_le, gen_p_limbs } from '../utils'
import assert from 'assert'

export async function smvp(
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
    const threads = 20

    // Misc parameters
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0    

    // Request GPU device
    const device = await get_device()
    const limits = device.limits

    console.log("limits is: ", limits)

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
        // Perform Sparse-Matrix Tranpose and SMVP
        await smvp_gpu(device, csr_sparse_matrices[i], num_words, word_size, p, n0, params.r, params.rinv)
    }
    device.destroy()
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
      for (let j = Math.ceil(lambda / s); j > 0; j--) {
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

export async function smvp_gpu(
    device: GPUDevice,
    csr_sm: CSRSparseMatrix,
    num_words: number,
    word_size: number,
    p: bigint,
    n0: bigint,
    r: bigint,
    rinv: bigint,
) {
    const csr_sparse_matrix_transposed = await csr_sm.transpose()

    const start_1 = Date.now()
    
    const vector_smvp: bigint[] = Array(csr_sparse_matrix_transposed.row_ptr.length - 1).fill(BigInt(1));
    const buckets_svmp: ExtPointType[] = await csr_sparse_matrix_transposed.smvp(vector_smvp)
    
    const elapsed_1 = Date.now() - start_1
    const output_points_non_mont_and_affine_cpu = buckets_svmp.map((x) => x.toAffine())
    console.log(`CPU took ${elapsed_1}ms`)

    // Convert BigIntPoint to montgomery form
    const points_with_mont_coords: BigIntPoint[] = []
    for (const pt of csr_sparse_matrix_transposed.data) {
        points_with_mont_coords.push(
            {
                x: fieldMath.Fp.mul(pt.x, r),
                y: fieldMath.Fp.mul(pt.y, r),
                t: fieldMath.Fp.mul(pt.t, r),
                z: fieldMath.Fp.mul(pt.z, r),
            }
        )
    }

    const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sparse_matrix_transposed.row_ptr)

    const NUM_ROWS = csr_sparse_matrix_transposed.row_ptr.length - 1

    let NUM_ROWS_GPU = 0
    let numBlocks = 1
    if (NUM_ROWS < 256) {
        NUM_ROWS_GPU = NUM_ROWS
    }
    else {
        const blockSize = 256;
        const totalThreads = NUM_ROWS
        numBlocks = Math.floor((totalThreads + blockSize - 1) / blockSize)
        NUM_ROWS_GPU = blockSize
    }

    // Define number of workgroups
    const num_x_workgroups = numBlocks; 

    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)

    // 1: Create a shader module with templating engine
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        smvp_shader,
        {
            word_size,
            num_words,
            n0,
            p_limbs,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
            NUM_ROWS_GPU,
        },
        {
            structs,
            bigint_functions,
            montgomery_product_funcs,
            field_functions,
            curve_parameters,
            curve_functions,
        },
    )
    const shaderModule = device.createShaderModule({
        code: shaderCode
    })

    const commandEncoder = device.createCommandEncoder();

    // 2: Create buffered memory accessible by the GPU memory space
    const output_buffer_length = NUM_ROWS * num_words * 4 * 4

    const start = Date.now()
    const output_storage_buffer = device.createBuffer({
        size: output_buffer_length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

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
        ]
    });

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [
            output_storage_buffer,
            row_ptr_storage_buffer,
            points_storage_buffer,
        ],
    )

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
        output_storage_buffer, // source
        0, // sourceOffset
        stagingBuffer, // destination
        0, // destinationOffset
        output_buffer_length,
    );

    // 8: Finish encoding commands and submit to GPU device command queue
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // 9: Map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        output_buffer_length,
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

    const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
        return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z)
    }

    // Convert output_points out of Montgomery coords
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

    // Convert output_points_non_mont into affine
    const output_points_non_mont_and_affine_gpu = output_points_non_mont.map((x) => x.toAffine())

    for (let i = 0; i < output_points_non_mont_and_affine_gpu.length; i ++) {
        assert(output_points_non_mont_and_affine_gpu[i].x === output_points_non_mont_and_affine_cpu[i].x)
        assert(output_points_non_mont_and_affine_gpu[i].y === output_points_non_mont_and_affine_cpu[i].y)
    }

    console.log("passed assertion checks!")
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

const create_bind_group = (device: GPUDevice,layout: GPUBindGroupLayout, buffers: GPUBuffer[]) => {
    const entries: any[] = []
    for (let i = 0; i < buffers.length; i ++) {
        entries.push({
            binding: i,
            resource: { buffer: buffers[i] }
        })
    }
    return device.createBindGroup({ layout, entries })
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
