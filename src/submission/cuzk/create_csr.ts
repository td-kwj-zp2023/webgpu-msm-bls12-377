import assert from 'assert'
import mustache from 'mustache'
import { BigIntPoint } from "../../reference/types"
import { CSRSparseMatrix } from '../matrices/matrices'; 
import { FieldMath } from "../../reference/utils/FieldMath";
import {
    gen_p_limbs,
    u8s_to_points,
    points_to_u8s_for_gpu,
    numbers_to_u8s_for_gpu,
    decompose_scalars,
    bigIntPointToExtPointType,
    u8s_to_numbers,
    compute_misc_params,
} from '../utils'
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { get_device, create_bind_group } from '../../submission/gpu'
import structs from '../wgsl/struct/structs.template.wgsl'
import bigint_funcs from '../wgsl/bigint/bigint.template.wgsl'
import field_funcs from '../wgsl/field/field.template.wgsl'
import ec_funcs from '../wgsl/curve/ec.template.wgsl'
import montgomery_product_funcs from '../wgsl/montgomery/mont_pro_product.template.wgsl'
import create_csr_shader from '../wgsl/create_csr.template.wgsl'

const fieldMath = new FieldMath()

export const precompute_with_cluster_method = (
    scalar_chunks: number[],
    row_idx: number,
    num_rows: number,
) => {
    assert(scalar_chunks.length % num_rows === 0)
    const num_cols = scalar_chunks.length / num_rows
    const clusters = new Map()

    // keep track of each cluster
    for (let i = 0; i < num_cols; i ++ ) {
        const pt_idx = row_idx * num_cols + i
        const chunk = scalar_chunks[pt_idx]

        // skip 0s
        if (chunk === 0) {
            continue
        }

        const g = clusters.get(chunk)
        if (g == undefined) {
            clusters.set(chunk, [pt_idx])
        } else {
            g.push(pt_idx)
            clusters.set(chunk, g)
        }
    }

    const cluster_start_indices: number[] = [0]
    const cluster_end_indices: number[] = []
    const new_point_indices: number[] = []

    for (const chunk of clusters.keys()) {
        const cluster = clusters.get(chunk)
        // append single-item clusters
        if (cluster.length === 1) {
            new_point_indices.push(cluster[0])
        } else {
            for (const c of cluster) {
                new_point_indices.unshift(c)
            }
        }
    }

    // populate cluster_start_indices and cluster_end_indices
    let prev_chunk = scalar_chunks[new_point_indices[0]]
    for (let i = 1; i < new_point_indices.length; i ++) {
        const s = scalar_chunks[new_point_indices[i]]
        if (prev_chunk != scalar_chunks[new_point_indices[i]]) {
            cluster_end_indices.push(i)
            cluster_start_indices.push(i)
        }
        prev_chunk = s
    }

    // the final cluster_end_index
    cluster_end_indices.push(new_point_indices.length)

    return { new_point_indices, cluster_start_indices, cluster_end_indices }
}


export function pre_aggregate<P> (
    points: P[],
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
    cluster_end_indices: number[],
    add_func: (a: P, b:P) => P = (a: any, b: any) => a + b,
)  {
    const new_points: any[] = []
    const new_scalar_chunks: number[] = []
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        const end_idx = cluster_end_indices[i]

        let acc = points[new_point_indices[start_idx]]
        for (let j = start_idx + 1; j < end_idx; j ++) {
            acc = add_func(acc, points[new_point_indices[j]])
        }

        new_points.push(acc)
        new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
    }

    return { new_points, new_scalar_chunks }
}

export const all_precomputation = (
    points: any[],
    scalar_chunks: number[],
    num_rows: number,
) => {
    let all_new_point_indices: number[] = []
    let all_cluster_start_indices: number[] = []
    let all_cluster_end_indices: number[] = []

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices, cluster_end_indices } = 
                precompute_with_cluster_method(scalar_chunks, row_idx, num_rows)

        for (let i = 0; i < cluster_start_indices.length; i ++) {
            cluster_start_indices[i] += all_new_point_indices.length
            cluster_end_indices[i] += all_new_point_indices.length
        }

        all_new_point_indices = all_new_point_indices.concat(new_point_indices)
        all_cluster_start_indices = all_cluster_start_indices.concat(cluster_start_indices)
        all_cluster_end_indices = all_cluster_end_indices.concat(cluster_end_indices)
    }
    return {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
    }
}

export const create_csr_cpu = (
    points: any[],
    scalar_chunks: number[],
    num_rows: number,
) => {
    const {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
    } = all_precomputation(points, scalar_chunks, num_rows)

    const { new_points, new_scalar_chunks } = pre_aggregate(
        points,
        scalar_chunks,
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
        (a: ExtPointType, b: ExtPointType) => a.add(b),
    )
    const row_ptr: number[] = all_cluster_start_indices
    row_ptr.push(all_new_point_indices.length)

    return new CSRSparseMatrix(new_points, new_scalar_chunks, row_ptr)
}

export const create_csr_sms_gpu = async (
    points: ExtPointType[],
    decomposed_scalars: number[][],
    num_rows: number,
    device: GPUDevice,
    points_storage_buffer: GPUBuffer,
    p: bigint,
    n0: bigint,
    rinv: bigint,
    num_words: number,
    word_size: number,
) => {
    const p_limbs = gen_p_limbs(p, num_words, word_size)
    const shaderCode = mustache.render(
        create_csr_shader,
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

    const shaderModule = device.createShaderModule({ code: shaderCode })
    const csr_sms_gpu: CSRSparseMatrix[] = []
    for (const scalar_chunks of decomposed_scalars) {
        const csr_sm = await create_csr_gpu(
            points,
            scalar_chunks,
            num_rows,
            device,
            shaderModule,
            points_storage_buffer,
            p,
            n0,
            rinv,
            num_words,
            word_size,
        )
        csr_sms_gpu.push(csr_sm)
    }

    return csr_sms_gpu
}

export const create_csr_gpu = async (
    points: ExtPointType[],
    scalar_chunks: number[],
    num_rows: number,
    device: GPUDevice,
    shaderModule: GPUShaderModule,
    points_storage_buffer: GPUBuffer,
    p: bigint,
    n0: bigint,
    rinv: bigint,
    num_words: number,
    word_size: number,
): Promise<CSRSparseMatrix> => {
    const {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
    } = all_precomputation(points, scalar_chunks, num_rows)
    const num_x_workgroups = 256

    const scalar_chunks_bytes = numbers_to_u8s_for_gpu(scalar_chunks)
    const new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices)
    const cluster_start_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_start_indices)
    const cluster_end_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_end_indices)

    const scalar_chunks_storage_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(scalar_chunks_storage_buffer, 0, scalar_chunks_bytes)

    const new_point_indices_storage_buffer = device.createBuffer({
        size: new_point_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(new_point_indices_storage_buffer, 0, new_point_indices_bytes)

    const cluster_start_indices_storage_buffer = device.createBuffer({
        size: cluster_start_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(cluster_start_indices_storage_buffer, 0, cluster_start_indices_bytes)

    const cluster_end_indices_storage_buffer = device.createBuffer({
        size: cluster_end_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
    device.queue.writeBuffer(cluster_end_indices_storage_buffer, 0, cluster_end_indices_bytes)

    const num_new_points = all_cluster_start_indices.length

    // Output buffers
    const new_points_storage_buffer = device.createBuffer({
        size: num_new_points * 320,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const new_scalar_chunks_storage_buffer = device.createBuffer({
        size: cluster_start_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    })

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [
            points_storage_buffer,
            scalar_chunks_storage_buffer,
            new_point_indices_storage_buffer,
            cluster_start_indices_storage_buffer,
            cluster_end_indices_storage_buffer,
            new_points_storage_buffer,
            new_scalar_chunks_storage_buffer,
        ],
    )

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    })

    const start = Date.now()

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
    const new_points_non_mont: any[] = []
    for (const pt of new_points) {
        const non = fieldMath.createPoint(
            fieldMath.Fp.mul(pt.x, rinv),
            fieldMath.Fp.mul(pt.y, rinv),
            fieldMath.Fp.mul(pt.t, rinv),
            fieldMath.Fp.mul(pt.z, rinv),
        )
        new_points_non_mont.push(non.toAffine())
    }

    const row_ptr: number[] = all_cluster_start_indices
    row_ptr.push(all_new_point_indices.length)
    return new CSRSparseMatrix(new_points_non_mont, new_scalar_chunks, row_ptr)

    //const data: ExtPointType[] = []
    //const col_idx: number[] = []
    //const row_ptr: number[] = []
    //return new CSRSparseMatrix(data, col_idx, row_ptr)
}

export async function create_csr_sparse_matrices_from_points_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const num_rows = 8
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const csr_sms = await create_csr_sparse_matrices_from_points(points, scalars, num_rows)
    console.log(csr_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function create_csr_sparse_matrices_from_points(
    points: ExtPointType[],
    scalars: bigint[],
    num_rows: number,
): Promise<CSRSparseMatrix[]> {
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(points.length % num_rows === 0)
    assert(points.length === scalars.length)

    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    // Compute CSR sparse matrices in CPU
    const expected_csr_sms: CSRSparseMatrix[] = []
    for (const scalar_chunks of decomposed_scalars) {
        const csr_sm = create_csr_cpu(
            points,
            scalar_chunks,
            num_rows,
        )
        expected_csr_sms.push(csr_sm)
    }

    const fieldMath = new FieldMath()
    const p = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')
    const params = compute_misc_params(p, word_size)
    const r = params.r
    const n0 = params.n0
    const rinv = params.rinv

    // Convert points to Montgomery coordinates
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

    const device = await get_device()
    // Store all the points in a GPU buffer
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);

    const csr_sms_gpu = await create_csr_sms_gpu(
        points,
        decomposed_scalars,
        num_rows,
        device,
        points_storage_buffer,
        p,
        n0,
        rinv,
        num_words,
        word_size,
    )

    assert(csr_sms_gpu.length === expected_csr_sms.length)
    for (let i = 0; i < expected_csr_sms.length; i ++) {
        try {
            assert(csr_sms_gpu[i].data.length === expected_csr_sms[i].data.length)
            assert(csr_sms_gpu[i].col_idx.length === expected_csr_sms[i].col_idx.length)
            assert(csr_sms_gpu[i].row_ptr.length === expected_csr_sms[i].row_ptr.length)

            for (let j = 0; j < expected_csr_sms[i].data.length; j ++) {
                assert(csr_sms_gpu[i].data[j].x === expected_csr_sms[i].data[j].x)
                assert(csr_sms_gpu[i].data[j].y === expected_csr_sms[i].data[j].y)
            }
            for (let j = 0; j < expected_csr_sms[i].col_idx.length; j ++) {
                assert(csr_sms_gpu[i].col_idx[j] === expected_csr_sms[i].col_idx[j])
            }
            for (let j = 0; j < expected_csr_sms[i].row_ptr.length; j ++) {
                assert(csr_sms_gpu[i].row_ptr[j] === expected_csr_sms[i].row_ptr[j])
            }
        } catch {
            console.log('assert fail at', i)
            break
        }
    }

    return csr_sms_gpu
}
