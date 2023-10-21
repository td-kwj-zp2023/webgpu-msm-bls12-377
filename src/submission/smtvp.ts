import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix, CSRSparseMatrix } from './matrices/matrices'; 
import store_point_at_infinity_shader from '../submission/wgsl/store_point_at_infinity.template.wgsl'
import smtvp_shader from '../submission/wgsl/smtvp.template.wgsl'
import bigint_struct from '../submission/wgsl/structs/bigint.template.wgsl'
import bigint_funcs from '../submission/wgsl/bigint.template.wgsl'
import montgomery_product_funcs from '../submission/wgsl/montgomery_product.template.wgsl'
import { compute_misc_params, u8s_to_points, points_to_u8s_for_gpu, numbers_to_u8s_for_gpu, gen_p_limbs, to_words_le } from './utils'
import { ExtPointType } from "@noble/curves/abstract/edwards";
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
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const word_size = 13
    const params = compute_misc_params(p, word_size)
    const n0 = params.n0
    const num_words = params.num_words
    assert(num_words === 20)

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
        const output_storage_buffer = await store_point_at_infinity_shader_gpu(device, max_col_idx, num_words, word_size, p, params.r)

        // Shader 2: perform SMTVP
        await smtvp_gpu(device, csr_sparse_matrices[i], num_words, word_size, p, n0, params.r, params.rinv, output_storage_buffer)
        break
    }

    return { x: BigInt(1), y: BigInt(0) }
}

export async function store_point_at_infinity_shader_gpu(
    device: GPUDevice,
    max_col_idx: number,
    num_words: number,
    word_size: number,
    p: bigint,
    r: bigint,
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

    const x_mont = fieldMath.Fp.mul(ZERO_POINT.ex, r)
    const y_mont = fieldMath.Fp.mul(ZERO_POINT.ey, r)
    const t_mont = fieldMath.Fp.mul(ZERO_POINT.et, r)
    const z_mont = fieldMath.Fp.mul(ZERO_POINT.ez, r)

    const mont_zero = {
        x: x_mont,
        y: y_mont,
        t: t_mont,
        z: z_mont,
    }

    const data_as_uint8s = new Uint8Array(data)

    stagingBuffer.unmap();

    const elapsed = Date.now() - start
    console.log(`GPU took ${elapsed}ms`)

    // TODO: skip this check in production
    for (const point of u8s_to_points(data_as_uint8s, num_words, word_size)) {
        assert(point.x === mont_zero.x)
        assert(point.y === mont_zero.y)
        assert(point.t === mont_zero.t)
        assert(point.z === mont_zero.z)
    }
    return output_storage_buffer
}

export async function smtvp_gpu(
    device: GPUDevice,
    csr_sm: CSRSparseMatrix,
    num_words: number,
    word_size: number,
    p: bigint,
    n0: bigint,
    r: bigint,
    rinv: bigint,
    previous_output_buffer: GPUBuffer,
) {
    const num_x_workgroups = 256;

    const max_col_idx = Math.max(...csr_sm.col_idx)
    const output_buffer_length = max_col_idx * num_words * 4 * 4

    const col_idx_bytes = numbers_to_u8s_for_gpu(csr_sm.col_idx)
    const row_ptr_bytes = numbers_to_u8s_for_gpu(csr_sm.row_ptr)
   
    const fieldMath = new FieldMath();

    // Convert each point coordinate for each point in the CSR matrix into
    // Montgomery form
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
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)

    const num_rows = csr_sm.row_ptr.length - 1

    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        smtvp_shader,
        {
            num_words,
            word_size,
            num_rows,
            n0,
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
    // shader. The values should each be the point at infinity in Montgomery
    // form (x: 0, y: r, t: 0, z: r).
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

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, output_buffer_length)
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();

    const elapsed = Date.now() - start

    console.log(`GPU took ${elapsed}ms`)

    const data_as_uint8s = new Uint8Array(data)

    const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
        return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z)
    }

    // Perform SMTVP in CPU
    const vec: bigint[] = []
    for (let i = 0; i < csr_sm.row_ptr.length - 1; i ++) {
        vec.push(BigInt(1))
    }
    const smtvp_result = await csr_sm.smtvp(vec, fieldMath)
    const smtvp_result_affine = smtvp_result.map((x) => x.toAffine())
    console.log(smtvp_result_affine)

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
    console.log(output_points_non_mont_and_affine)

    debugger
    assert(smtvp_result_affine.length === output_points_non_mont_and_affine.length)


    //for (let i = 0; i < smtvp_result_affine.length; i ++) {
        //assert(smtvp_result_affine[i].x === output_points_non_mont_and_affine[i].x)
        //assert(smtvp_result_affine[i].y === output_points_non_mont_and_affine[i].y)
    //}
    return

    console.log('0th output point from gpu:', output_points[0])

    const output_points_as_affine: any[] = []
    for (const output_point of output_points) {
        const pt = fieldMath.createPoint(
            fieldMath.Fp.mul(output_point.x, rinv),
            fieldMath.Fp.mul(output_point.y, rinv),
            fieldMath.Fp.mul(output_point.t, rinv),
            fieldMath.Fp.mul(output_point.z, rinv),
        )
        pt.assertValidity()
        output_points_as_affine.push(pt.toAffine())
    }
    console.log('0th output point from gpu as affine:', output_points_as_affine[0])
    //const extPointTypeToBigIntPoint = (e: ExtPointType): BigIntPoint => {
        //return {
            //x: e.ex,
            //y: e.ey,
            //t: e.et,
            //z: e.ez,
        //}
    //}

    // Add the points using noble-curves
    const pt = bigIntPointToExtPointType(csr_sm.data[0])
    const expected_orig = pt.add(pt)
    expected_orig.assertValidity()

    // add_points in Typescript for line-by-line debugging
    const expected_mont = add_points(
        points_with_mont_coords[0],
        points_with_mont_coords[0],
        p,
        r,
        rinv,
        fieldMath,
    )

    // Convert from Montgomery form
    const expected_mont_to_orig = fieldMath.createPoint(
        fieldMath.Fp.mul(expected_mont.ex, rinv),
        fieldMath.Fp.mul(expected_mont.ey, rinv),
        fieldMath.Fp.mul(expected_mont.et, rinv),
        fieldMath.Fp.mul(expected_mont.ez, rinv),
    )
    expected_mont_to_orig.assertValidity()

    console.log('noble affine output:', expected_orig.toAffine())
    console.log('ts affine output:   ', expected_mont_to_orig.toAffine())

    assert(expected_orig.x === expected_mont_to_orig.x)
    assert(expected_orig.y === expected_mont_to_orig.y)
    assert(expected_orig.x === output_points_as_affine[0].x)
    assert(expected_orig.y === output_points_as_affine[0].y)
}

export const add_points = (
    p1: BigIntPoint,
    p2: BigIntPoint,
    p: bigint,
    r: bigint,
    rinv: bigint,
    fieldMath: FieldMath,
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

    const EDWARDS_D = fieldMath.Fp.mul(BigInt(3021), r)
    const c = montgomery_product(EDWARDS_D, t2)

    const p1z = p1.z
    const p2z = p2.z
    const d = montgomery_product(p1z, p2z)

    const xpy = fr_add(p1x, p1y)
    const xpy2 = fr_add(p2x, p2y)
    let e = montgomery_product(xpy, xpy2)

    e = fr_sub(e, a)
    e = fr_sub(e, b)

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

