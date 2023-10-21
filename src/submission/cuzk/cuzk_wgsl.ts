import { BigIntPoint } from "../../reference/types";
import { bigints_to_u8_for_gpu } from '../../submission/utils'
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { CSRSparseMatrix, ELLSparseMatrix, fieldMath } from '../matrices/matrices';
import store_point_at_infinity_shader from '../wgsl/store_point_at_infinity.template.wgsl'
import transpose_shader from '../wgsl/tranpose.template.wgsl'
import bigint_struct from '../wgsl/structs.template.wgsl'
import bigint_funcs from '../wgsl/bigint.template.wgsl'
import ec_funcs from '../wgsl/ec.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery_product.template.wgsl'
import { u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu, compute_misc_params, gen_p_limbs } from '../utils'
import assert from 'assert'
import { wordSize } from "bn.js";
import exp from "constants";
import { exit } from "process";

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

    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const word_size = 13
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    assert(num_words === 20)

    // Request GPU device
    const device = await get_device()

    // Generate CSR sparse matrices
    const csr_sparse_matrices = await gen_csr_sparse_matrices(
        baseAffinePoints,
        scalars,
        lambda,
        s,
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
    console.log("entered tranpose_gpu method!")
    
    console.log("csr data is: ", csr_sm.data)

    const tester: BigIntPoint = { 
        x: BigInt(0),
        y: BigInt(1),
        t: BigInt(0),
        z: BigInt(1),
    }

    // csr_sm.data[0] = tester
    csr_sm.data[1] = tester

    console.log("0 !", csr_sm.data[0])
    console.log("1 !", csr_sm.data[1])

    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;
    console.log("zero_point is: ", ZERO_POINT)
    const test_cpu_point = fieldMath.createPoint(csr_sm.data[0].x, csr_sm.data[0].y, csr_sm.data[0].t, csr_sm.data[0].z)
    const test_cpu_point_2 = fieldMath.createPoint(csr_sm.data[1].x, csr_sm.data[1].y, csr_sm.data[1].t, csr_sm.data[1].z)
    console.log("test_cpu_point: ", test_cpu_point)
    console.log("test_cpu_point_2: ", test_cpu_point_2)
    const add_point = test_cpu_point.add(test_cpu_point_2)
    console.log("test cpu point add ! ", add_point)
    console.log("test cpu point add affine! ", add_point.toAffine())

    // convert points to montgomery form
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

    console.log("????? points_with_mont_coords: ", points_with_mont_coords)

    // Define number of workgroups
    const num_x_workgroups = 1;

    console.log("csr_sm.col_idx; ", csr_sm.col_idx)

    console.log("csr_sm.data, is: ", csr_sm.data)

    const col_idx_bytes = numbers_to_u8s_for_gpu(csr_sm.col_idx)
    const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sm.row_ptr)
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const num_rows = csr_sm.row_ptr.length - 1

    console.log("points_bytes is: ", points_bytes)
    const points_bytes_points = new Uint8Array(points_bytes)
    const points_new = u8s_to_points(points_bytes_points, num_words, word_size)
    console.log("!!!!!!!!!!! points_new is: ", points_new)

    // 1: Create a shader module with templating engine
    // const shaderCode = mustache.render(
    //     transpose_shader,
    //     {
    //         num_words,
    //         num_rows,
    //     },
    //     {
    //         bigint_struct,
    //     },
    // )
    // const shaderCode = mustache.render(transpose_shader, { num_words })
    // const shaderModule = device.createShaderModule({
    //     code: shaderCode
    // })
    const p_limbs = gen_p_limbs(p, num_words, word_size)

    
    const shaderCode = mustache.render(
        transpose_shader,
        {
            num_words,
            word_size,
            num_rows,
            max_col_idx,
            n0,
            p_limbs,
            two_pow_word_size: 2 ** word_size,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
        },
        {
            bigint_struct,
            bigint_funcs,
            montgomery_product_funcs,
            ec_funcs
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

    // const col_idx_storage_buffer = device.createBuffer({
    //     size: col_idx_bytes.length,
    //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    // });
    // device.queue.writeBuffer(col_idx_storage_buffer, 0, col_idx_bytes);

    // const row_ptr_storage_buffer = device.createBuffer({
    //     size: row_ptr_bytes.length,
    //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    // });
    // device.queue.writeBuffer(row_ptr_storage_buffer, 0, row_ptr_bytes);

    console.log("points_bytes.length is: ", points_bytes.length)

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
            // { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            // { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ]
    });

    // Bind Group represents the actual input/output data for the shader, and associate with GPU buffers 
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: output_storage_buffer } },
            // { binding: 1, resource: { buffer: col_idx_storage_buffer } },
            // { binding: 2, resource: { buffer: row_ptr_storage_buffer } },
            { binding: 1, resource: { buffer: points_storage_buffer } },
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

    console.log("points_bytes.length is: ", points_bytes.length)

    // Returns an ArrayBuffer with the contents of the GPUBuffer in the given mapped range.
    const JSArrayBuffer = stagingBuffer.getMappedRange(0, output_buffer_length)
    const data = JSArrayBuffer.slice(0);

    stagingBuffer.unmap();

    // End Timer
    const elapsed = Date.now() - start
    console.log(`GPU took ${elapsed}ms`)

    // Print results
    const data_as_uint8s = new Uint8Array(data)

    const output_points = u8s_to_points(data_as_uint8s, num_words, word_size)
    console.log('! 0th output point from gpu:', output_points[0])
    // const abr_gpu = fieldMath.Fp.mul(points[0].z, r)
    // console.log("! abr gpu is: ", abr_gpu)
    const gpu_point = fieldMath.createPoint(output_points[0].x, output_points[0].y, output_points[0].t, output_points[0].z)
    const gpu_point_affine = gpu_point.toAffine()

    // Algorithm here:
    const expected = add_points(points_with_mont_coords[0], points_with_mont_coords[1], p, rinv, r)
    const expected_affine = expected.toAffine()
    
    console.log('result:', gpu_point)
    console.log('expected:', expected)

    console.log('result affine:', gpu_point_affine)
    console.log('expected affine:', expected_affine)

    assert(expected_affine.x === gpu_point_affine.x)
    assert(expected_affine.y === gpu_point_affine.y)
    console.log("assert passed!")

    const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246')

    const ab = fieldMath.Fp.mul(points_with_mont_coords[0].x, points_with_mont_coords[0].x)
    console.log("! ab is: ", ab)
    const abr = fieldMath.Fp.mul(ab, rinv)
    console.log("! abr is: ", abr)

    const xr = points_with_mont_coords[0].x
    assert(((x * x * r) % p) == ((xr * x) % p))

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

    const fr_add = fieldMath.Fp.add
    const fr_sub = fieldMath.Fp.sub

    const p1x = p1.x
    const p2x = p2.x
    const a = montgomery_product(p1x, p2x)

    const p1y = p1.y
    const p2y = p2.y
    const b = montgomery_product(p1y, p2y)

    const p1t = p1.t
    const p2t = p2.t
    const t2 = montgomery_product(p1t, p2t)

    const EDWARDS_D = BigInt(3021) * r
    const c = montgomery_product(EDWARDS_D, t2)

    const p1z = p1.z
    const p2z = p2.z
    const d = montgomery_product(p1z, p2z)

    const p1_added = fr_add(p1x, p1y)

    const p2_added = fr_add(p2x, p2y)

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

/*
  var a = field_multiply(p1.x, p2.x);
  var b = field_multiply(p1.y, p2.y);
  var c = field_multiply(EDWARDS_D, field_multiply(p1.t, p2.t));
  var d = field_multiply(p1.z, p2.z);
  var p1_added = field_add(p1.x, p1.y);
  var p2_added = field_add(p2.x, p2.y);
  var e = field_multiply(field_add(p1.x, p1.y), field_add(p2.x, p2.y));
  e = field_sub(e, a);
  e = field_sub(e, b);
  var f = field_sub(d, c);
  var g = field_add(d, c);
  var a_neg = mul_by_a(a);
  var h = field_sub(b, a_neg);
  var added_x = field_multiply(e, f);
  var added_y = field_multiply(g, h);
  var added_t = field_multiply(e, h);
  var added_z = field_multiply(f, g);
  return Point(added_x, added_y, added_t, added_z);
  */