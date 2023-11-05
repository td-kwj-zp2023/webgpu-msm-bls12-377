import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { FieldMath } from "../reference/utils/FieldMath";
import { decompose_scalars, bigIntPointToExtPointType, compute_misc_params } from './utils'
import { get_device, create_bind_group } from '../submission/gpu'
import {
    gen_p_limbs,
    u8s_to_points,
    points_to_u8s_for_gpu,
    numbers_to_u8s_for_gpu,
    u8s_to_numbers,
} from './utils'
import {
    prep_for_cluster_method,
    pre_aggregate_cpu,
} from './create_ell'
import assert from 'assert'
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";

import create_ell_shader from './wgsl/create_ell.template.wgsl'
import structs from './wgsl/struct/structs.template.wgsl'
import bigint_funcs from './wgsl/bigint/bigint.template.wgsl'
import field_funcs from './wgsl/field/field.template.wgsl'
import ec_funcs from './wgsl/curve/ec.template.wgsl'
import montgomery_product_funcs from './wgsl/montgomery/mont_pro_product.template.wgsl'

const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
const word_size = 13

export async function create_ell_sparse_matrices_from_points_gpu_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const num_threads = 8
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const params = compute_misc_params(p, word_size)
    const r = params.r

    // Convert points to Montgomery coordinates
    // In the actual impl, this should be done inside the shader
    const points_with_mont_coords: BigIntPoint[] = []
    for (const pt of points) {
        points_with_mont_coords.push(
            {
                x: fieldMath.Fp.mul(pt.ex, r),
                y: fieldMath.Fp.mul(pt.ey, r),
                t: fieldMath.Fp.mul(pt.et, r),
                z: fieldMath.Fp.mul(pt.ez, r),
            }
        )
    }

    const ell_sms = await create_ell_sparse_matrices_from_points_gpu(points_with_mont_coords, scalars, num_threads)
    console.log(ell_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function create_ell_sparse_matrices_from_points_gpu(
    points_with_mont_coords: BigIntPoint[],
    scalars: bigint[],
    num_threads: number,
): Promise<ELLSparseMatrix[]> {
    const device = await get_device()

    const ell_sms: ELLSparseMatrix[] = []
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(points_with_mont_coords.length % num_threads === 0)
    assert(points_with_mont_coords.length === scalars.length)

    const params = compute_misc_params(p, word_size)
    const num_words = params.num_words

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);

    const ell_sms_serial: ELLSparseMatrix[] = []

    for (const scalar_chunks of decomposed_scalars) {
        const ell_sm = await create_ell_gpu(
            device,
            points_storage_buffer,
            points_with_mont_coords,
            scalar_chunks,
            num_threads,
            params,
        )
        ell_sms_serial.push(ell_sm)
    }

    return ell_sms
}

const fieldMath = new FieldMath()

// Create an ELL sparse matrix from all the points of the MSM and a set of
// scalar chunks
export async function create_ell_gpu(
    device: GPUDevice,
    points_storage_buffer: GPUBuffer,
    points_with_mont_coords: BigIntPoint[],
    scalar_chunks: number[],
    num_rows: number,
    params: any,
) {
    const num_words = params.num_words
    const n0 = params.n0
    const rinv = params.rinv

    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    // This shader should compute a row of the sparse matrix.
    // new_point_indices and cluster_start_indices should be computed in the CPU
    //
    // Input buffers:
    //   - points (with coords in Mont form)
    //   - scalar_chunks
    //   - new_point_indices
    //   - cluster_start_indices, up to stop_at
    //
    //   r = num_rows
    //   j = n / r
    //   s = stop_at
    //   s <= j
    //
    //   points: [P0, ..., Pn]
    //   scalar_chunks: [C0, ..., Cn]
    //   new_point_indices: [
    //                        N0_0, ..., N0_j,
    //                        ...,
    //                        Nr_0, ..., Nr_j
    //                      ]
    //   cluster_start_indices: [
    //                            S0_0, ..., S0_s,
    //                            ...,
    //                            Sr_0, ..., Sr_s,
    //                          ]
    //
    // Output buffers:
    //   - new_points
    //   - new_scalar_chunks
    //
    //   new_points: [
    //                 newP0_0, ..., newP0_j, 
    //                 ...,
    //                 newPr_0, ..., newPr_j, 
    //               ]

    const all_new_point_indices = []
    const all_cluster_start_indices = []

    const expected_new_points = []
    const expected_scalar_chunks = []

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices } = prep_for_cluster_method(
            scalar_chunks,
            row_idx,
            num_rows,
        )

        //const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
            //points, 
            //scalar_chunks,
            //new_point_indices,
            //cluster_start_indices,
        //)

        //expected_new_points.push(new_points)
        //expected_scalar_chunks.push(new_scalar_chunks)

        // Append the final end_idx
        cluster_start_indices.push(points_with_mont_coords.length / num_rows)

        for (const a of new_point_indices) {
            all_new_point_indices.push(a)
        }

        let prev = cluster_start_indices[cluster_start_indices.length - 1]
        let lim = cluster_start_indices.length - 2
        for (; lim >= 0; lim --) {
            if (prev - cluster_start_indices[lim] > 1) {
                break
            }
            prev = cluster_start_indices[lim]
        }
        
        for (let j = 0; j < lim + 2; j ++) {
            all_cluster_start_indices.push(cluster_start_indices[j])
        }
    }

    // Convert inputs to bytes
    const scalar_chunks_bytes = numbers_to_u8s_for_gpu(scalar_chunks)
    const all_new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices)
    const all_cluster_start_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_start_indices)

    // Each x workgroup can operate on workgroup_size values at a time.
    const workgroup_size = 256
    const max_threads = workgroup_size * 256
    const num_new_points = all_cluster_start_indices.length

    const num_invocations = Math.ceil(num_new_points / max_threads)

    const scalar_chunks_storage_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(scalar_chunks_storage_buffer, 0, scalar_chunks_bytes);

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    })

    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        create_ell_shader,
        {
            num_words,
            word_size,
            n0,
            mask: BigInt(2) ** BigInt(word_size) - BigInt(1),
            two_pow_word_size: BigInt(2) ** BigInt(word_size),
            p_limbs,
        },
        {
            structs,
            bigint_funcs,
            field_funcs,
            ec_funcs,
            montgomery_product_funcs,
        }
    )

    const shaderModule = device.createShaderModule({
        code: shaderCode
    })

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    const do_gpu_invocation = async (
        num_x_workgroups: number,
        invocation_new_point_indices_bytes: Uint8Array,
        invocation_cluster_start_indices_bytes: Uint8Array,
    ) => {
        const start = Date.now()
        const invocation_new_point_indices_storage_buffer = device.createBuffer({
            size: invocation_new_point_indices_bytes.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        })

        const invocation_cluster_start_indices_storage_buffer = device.createBuffer({
            size: invocation_cluster_start_indices_bytes.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        })

        device.queue.writeBuffer(invocation_new_point_indices_storage_buffer, 0, invocation_new_point_indices_bytes);
        device.queue.writeBuffer(invocation_cluster_start_indices_storage_buffer, 0, invocation_cluster_start_indices_bytes);

        const num_new_points = invocation_new_point_indices_bytes.length / 4

        // Output buffers
        const new_points_storage_buffer = device.createBuffer({
            size: num_new_points * 320,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        })

        const new_scalar_chunks_storage_buffer = device.createBuffer({
            size: invocation_cluster_start_indices_bytes.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        })

        const bindGroup = create_bind_group(
            device, 
            bindGroupLayout,
            [
                points_storage_buffer,
                scalar_chunks_storage_buffer,
                invocation_new_point_indices_storage_buffer,
                invocation_cluster_start_indices_storage_buffer,
                new_points_storage_buffer,
                new_scalar_chunks_storage_buffer,
            ],
        )

        const commandEncoder = device.createCommandEncoder()
        const passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(computePipeline)
        passEncoder.setBindGroup(0, bindGroup)
        passEncoder.dispatchWorkgroups(num_x_workgroups)
        passEncoder.end()

        const new_points_staging_buffer = device.createBuffer({
            size: new_points_storage_buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const new_scalar_chunks_staging_buffer = device.createBuffer({
            size: new_scalar_chunks_storage_buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        commandEncoder.copyBufferToBuffer(
            new_points_storage_buffer,
            0,
            new_points_staging_buffer,
            0,
            new_points_storage_buffer.size
        );
        commandEncoder.copyBufferToBuffer(
            new_scalar_chunks_storage_buffer,
            0,
            new_scalar_chunks_staging_buffer,
            0,
            new_scalar_chunks_storage_buffer.size
        );

        device.queue.submit([commandEncoder.finish()]);

        // map staging buffers to read results back to JS
        await new_points_staging_buffer.mapAsync(
            GPUMapMode.READ,
            0,
            new_points_storage_buffer.size
        );
        await new_scalar_chunks_staging_buffer.mapAsync(
            GPUMapMode.READ,
            0,
            new_scalar_chunks_storage_buffer.size
        );

        const np = new_points_staging_buffer.getMappedRange(0, new_points_staging_buffer.size)
        const new_points_data = np.slice(0)
        new_points_staging_buffer.unmap()

        const ns = new_scalar_chunks_staging_buffer.getMappedRange(0, new_scalar_chunks_staging_buffer.size)
        const new_scalar_chunks_data = ns.slice(0)
        new_scalar_chunks_staging_buffer.unmap()

        const elapsed = Date.now() - start
        console.log(`GPU took ${elapsed}ms`)
        const new_scalar_chunks = u8s_to_numbers(new Uint8Array(new_scalar_chunks_data))
        const new_points = u8s_to_points(new Uint8Array(new_points_data), num_words, word_size)

        // Convert out of Mont form
        const new_points_non_mont: ExtPointType[] = []
        for (const pt of new_points) {
            const non = {
                x: fieldMath.Fp.mul(pt.x, rinv),
                y: fieldMath.Fp.mul(pt.y, rinv),
                t: fieldMath.Fp.mul(pt.t, rinv),
                z: fieldMath.Fp.mul(pt.z, rinv),
            }
            new_points_non_mont.push(bigIntPointToExtPointType(non, fieldMath))
        }

        return {
            new_points_non_mont,
            new_scalar_chunks,
        }
    }

    for (let invocation_idx = 0; invocation_idx < num_invocations; invocation_idx ++) {
        const num_points_for_this_invocation =
            invocation_idx === num_invocations - 1 ? 
                num_new_points % max_threads
                :
                max_threads

        const num_x_workgroups = num_points_for_this_invocation % 256 === 0 ?
            num_points_for_this_invocation / 256
            :
            Math.floor(num_points_for_this_invocation / 256) + 1

        const invocation_new_point_indices_bytes = all_new_point_indices_bytes.slice(
            invocation_idx * max_threads * 4,
            (invocation_idx * max_threads + num_points_for_this_invocation) * 4,
        )
        const invocation_cluster_start_indices_bytes = all_cluster_start_indices_bytes.slice(
            invocation_idx * max_threads,
            (invocation_idx * max_threads + num_points_for_this_invocation) * 4,
        )

        console.log(`Invocation ${invocation_idx}`)
        const { new_points_non_mont, new_scalar_chunks } =
            await do_gpu_invocation(
                num_x_workgroups,
                invocation_new_point_indices_bytes,
                invocation_cluster_start_indices_bytes,
            )
    }

    console.log('Note that converting bytes to BigInts and ExtPointTypes is inefficient')

    return new ELLSparseMatrix(data, col_idx, row_length)
}
