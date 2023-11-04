import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix } from './matrices/matrices'; 
import { get_device, create_bind_group } from '../submission/gpu'
import { BigIntPoint } from "../reference/types"
import {
    gen_p_limbs,
    u8s_to_points,
    compute_misc_params,
    points_to_u8s_for_gpu,
    numbers_to_u8s_for_gpu,
} from './utils'
import create_ell_shader from '../submission/wgsl/create_ell.template.wgsl'

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'
import mustache from 'mustache'

export const compute_new_scalar_chunks = (
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
) => {
    const new_scalar_chunks: number[] = []

    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        if (scalar_chunks[new_point_indices[start_idx]] !== 0) {
            new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
        }
    }
    return new_scalar_chunks
}

/*
 * @param: points All the input points of the MSM.
 * @param: new_point_indices The output of a prep function, such as prep_for_cluster_method
 * @param: cluster_start_indices The output of a prep function, such as prep_for_cluster_method
 * @return { new_points, new_scalar_chunks }: an array of points and associated scalar chunks
 * This function constructs a row of an ELL sparse matrix by pre-aggregating
 * points with the same scalar chunk.
 */
export const pre_aggregate_cpu = (
    points: ExtPointType[],
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
) => {
    const new_points: ExtPointType[] = []
    const new_scalar_chunks: number[] = []

    for (let i = 0; i < cluster_start_indices.length; i ++) {
        // Example: [0, 1, 4, 7]
        // Case 0: 0, 1
        // Case 1: 1, 4
        // Case 2: 4, 7
        // Case 3: 7, end
        
        const start_idx = cluster_start_indices[i]
        if (scalar_chunks[new_point_indices[start_idx]] === 0) {
            continue
        }

        let end_idx
        if (i === cluster_start_indices.length - 1) {
            // Case 3: we've reached the end of the array
            end_idx = new_point_indices.length
        } else {
            // Cases 1, 2: end_idx is the next index
            end_idx = cluster_start_indices[i + 1]
        }

        const p = points[new_point_indices[start_idx]]
        let acc = fieldMath.createPoint(p.ex, p.ey, p.et, p.ez)
        for (let j = start_idx + 1; j < end_idx; j ++) {
            const p = points[new_point_indices[j]]
            const pt = fieldMath.createPoint(p.ex, p.ey, p.et, p.ez)
            acc = acc.add(pt)
        }
        new_points.push(acc)
        new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
    }
    return { new_points, new_scalar_chunks }
}

/*
 * This "prep" method is meant to prepare inputs to assist the pre-aggregation
 * step. It sorts the scalar chunks, then calculates the cluster start indices.
 * e.g. if the scalar chunks are [3, 1, 3],
 * the sorted scalar chunks are [1, 3, 3], and
 * cluster_start_indices will be [0, 1, 2]
 * new_point_indices will be [1, 0, 2] 
 */
export const prep_for_sort_method = (
    scalar_chunks: number[],
    thread_idx: number,
    num_threads: number,
) => {
    assert(num_threads > 0)
    const pt_and_chunks = []
    const c = scalar_chunks.length / num_threads
    for (let i = 0; i < c; i ++) {
        const pt_idx = thread_idx * c + i
        pt_and_chunks.push([pt_idx, scalar_chunks[i]])
    }

    pt_and_chunks.sort((a: number[], b: number[]) => {
        if (a[1] > b[1]) { return 1 }
        else if (a[1] < b[1]) { return -1 }
        return 0
    })

    const cluster_start_indices = [0]
    let prev_chunk = pt_and_chunks[0][1]
    for (let k = 1; k < pt_and_chunks.length; k ++) {
        if (prev_chunk !== pt_and_chunks[k][1]) {
            cluster_start_indices.push(k)
        }
        prev_chunk = pt_and_chunks[k][1]
    }

    const new_point_indices = pt_and_chunks.map((x) => x[0])
    return { new_point_indices, cluster_start_indices }
}

/*
 * This "prep" method is meant to prepare inputs to assist the pre-aggregation
 * step. Instead of sorting the entire array, it just tallies the indices of
 * unique scalar chunks, then calculates the cluster start indices.
 * e.g. if the scalar chunks are [3, 1, 3],
 * the clustered scalar chunks are { 1: [1], 3: [0, 1]}
 * The clustered scalar chunks are then flattened, with clusters with more than
 * 1 element placed at the start of the array: [2, 0, 1]
 * cluster_start_indices will be [0, 1, 2]
 * new_point_indices will be [2, 0, 1] 
 *
 * This method is more efficient than prep_for_sort_method because the order of
 * the clusters does not matter. What does matter is that they are clustered.
 */
export const prep_for_cluster_method = (
    scalar_chunks: number[],
    thread_idx: number,
    num_threads: number,
) => {
    assert(num_threads > 0)
    const new_point_indices: number[] = []
    const cluster_start_indices: number[] = [0]
    const clusters = new Map()

    const c = scalar_chunks.length / num_threads
    for (let i = 0; i < c; i ++) {
        const pt_idx = thread_idx * c + i
        const chunk = scalar_chunks[pt_idx]
        const g = clusters.get(chunk)
        if (g == undefined) {
            clusters.set(chunk, [pt_idx])
        } else {
            g.push(pt_idx)
            clusters.set(chunk, g)
        }
    }

    for (const k of clusters.keys()) {
        const cluster = clusters.get(k)
        if (cluster.length === 1) {
            new_point_indices.push(cluster[0])
        } else {
            for (const c of cluster) {
                new_point_indices.unshift(c)
            }
        }
    }

    // Build cluster_start_indices
    let prev_chunk = scalar_chunks[new_point_indices[0]]
    for (let i = 1; i < new_point_indices.length; i ++) {
        if (prev_chunk !== scalar_chunks[new_point_indices[i]]) {
            cluster_start_indices.push(i)
        }
        prev_chunk = scalar_chunks[new_point_indices[i]]
    }

    return { new_point_indices, cluster_start_indices }
}

const fieldMath = new FieldMath()

export function create_ell(
    points: ExtPointType[],
    scalar_chunks: number[],
    num_threads: number,
) {
    const num_cols = scalar_chunks.length / num_threads
    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
        const { new_point_indices, cluster_start_indices } = prep_for_cluster_method(
            scalar_chunks,
            thread_idx,
            num_threads,
        )
        const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
            points, 
            scalar_chunks,
            new_point_indices,
            cluster_start_indices,
        )

        const pt_row: ExtPointType[] = []
        const idx_row: number[] = []

        assert(new_points.length === new_scalar_chunks.length)
        for (let j = 0; j < new_points.length; j ++) {
            assert(new_scalar_chunks[j] !== 0)
            pt_row.push(new_points[j])
            idx_row.push(new_scalar_chunks[j])
        }

        data.push(pt_row)
        col_idx.push(idx_row)
        row_length.push(pt_row.length)
    }

    return new ELLSparseMatrix(data, col_idx, row_length)
}

// Create an ELL sparse matrix from all the points of the MSM and a set of
// scalar chunks
export async function create_ell_gpu(
    points: ExtPointType[],
    scalar_chunks: number[],
    num_rows: number,
) {
    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []
    const num_x_workgroups = 256

    // This shader should compute a row of the sparse matrix.
    // new_point_indices and cluster_start_indices should be computed in the CPU
    //
    // Input buffers:
    //   - points
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
    //
    // Shader logic:
    //   gidx = global_id.x
    //   gidy = global_id.y
    //
    //   start_idx = cluster_start_indices[gidx * 2u]
    //   end_idx = cluster_start_indices[gidx * 2u] + 1
    //
    //   pt = points[gidy + new_point_indices[start_idx]]
    //
    //   for (i in range(start_idx + 1u, end_idx):
    //      pt = add_points(pt, points[gidy + new_point_indices[i]])
    //
    //   new_points[gidx] = pt
    //   new_scalar_chunks[gidx] = scalar_chunks[gidy + new_point_indices[start_idx]]

    const all_new_point_indices = []
    const all_cluster_start_indices = []
    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices } = prep_for_cluster_method(
            scalar_chunks,
            row_idx,
            num_rows,
        )
        // Append the final end_idx
        cluster_start_indices.push(points.length / num_rows)

        for (const a of new_point_indices) {
            all_new_point_indices.push(a)
        }
        for (const a of cluster_start_indices) {
            all_cluster_start_indices.push(a)
        }
    }

    const word_size = 13
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

    const params = compute_misc_params(p, word_size)
    const num_words = params.num_words
    const r = params.r
    //const n0 = params.n0
    //const rinv = params.rinv

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

    // Convert inputs to bytes
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const scalar_chunks_bytes = numbers_to_u8s_for_gpu(scalar_chunks)
    const all_new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices)
    const all_cluster_start_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_start_indices)

    const device = await get_device()
    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC // read-only
    });
    const scalar_chunks_storage_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC // read-only
    });
    const all_new_point_indices_storage_buffer = device.createBuffer({
        size: all_new_point_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC // read-only
    });
    const all_cluster_start_indices_storage_buffer = device.createBuffer({
        size: all_cluster_start_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC // read-only
    });

    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);
    device.queue.writeBuffer(scalar_chunks_storage_buffer, 0, scalar_chunks_bytes);
    device.queue.writeBuffer(all_new_point_indices_storage_buffer, 0, all_new_point_indices_bytes);
    device.queue.writeBuffer(all_cluster_start_indices_storage_buffer, 0, all_cluster_start_indices_bytes);

    // Output buffers
    const new_points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const new_scalar_chunks_storage_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    });

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [
            points_storage_buffer,
            scalar_chunks_storage_buffer,
            all_new_point_indices_storage_buffer,
            all_cluster_start_indices_storage_buffer,
            new_points_storage_buffer,
            new_scalar_chunks_storage_buffer,
        ],
    )

    const shaderCode = mustache.render(
        create_ell_shader,
        {
            //num_words,
            //word_size,
            //n0,
            //mask,
            //two_pow_word_size,
            //cost,
            //p_limbs,
        },
        {
            //structs,
            //bigint_functions,
            //curve_functions,
            //curve_parameters,
            //field_functions,
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

    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups)
    passEncoder.end()

    const new_points_staging_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const new_scalar_chunks_staging_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
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

    const np = new_points_staging_buffer.getMappedRange(0, new_points_storage_buffer.size)
    const new_points_data = np.slice(0)
    new_points_staging_buffer.unmap()

    const ns = new_points_staging_buffer.getMappedRange(0, new_points_storage_buffer.size)
    const new_scalar_chunks_data = ns.slice(0)
    new_points_staging_buffer.unmap()

    return new ELLSparseMatrix(data, col_idx, row_length)
}
