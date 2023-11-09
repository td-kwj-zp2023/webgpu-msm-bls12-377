import assert from 'assert'
import mustache from 'mustache'
import { ExtPointType } from "@noble/curves/abstract/edwards";
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
    const num_rows = 4
    const points = baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath))
    const params = compute_misc_params(p, word_size)
    const r = params.r

    const ell_sms = await create_ell_sparse_matrices_from_points_gpu(points, r, scalars, num_rows)
    //console.log(ell_sms)
    return { x: BigInt(0), y: BigInt(1) }
}

export async function create_ell_sparse_matrices_from_points_gpu(
    points: ExtPointType[],
    r: bigint,
    scalars: bigint[],
    num_rows: number,
): Promise<ELLSparseMatrix[]> {
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

    const ell_sms: ELLSparseMatrix[] = []
    const ell_sms_serial: ELLSparseMatrix[] = []

    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(points_with_mont_coords.length % num_rows === 0)
    assert(points_with_mont_coords.length === scalars.length)

    const params = compute_misc_params(p, word_size)
    const num_words = params.num_words

    // Decompose scalars
    //const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)
    const decomposed_scalars = [
        [33, 0, 2, 36, 11, 23, 16, 30, 36, 23, 16, 2, 32, 19, 18, 33, 34, 29, 31, 31, 36, 10, 6, 23, 24, 15, 6, 32, 29, 9, 0, 11, 19, 15, 27, 17, 33, 14, 36, 9, 25, 21, 7, 19, 13, 14, 30, 19, 16, 24, 32, 28, 22, 0, 23, 26, 14, 22, 14, 32, 4, 33, 1, 11]
    ]

    const device = await get_device()

    // Store all the points in a GPU buffer
    const points_bytes = points_to_u8s_for_gpu(points_with_mont_coords, num_words, word_size)
    const points_storage_buffer = device.createBuffer({
        size: points_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(points_storage_buffer, 0, points_bytes);

    // Serially iterate through each set of scalar chunks
    for (const scalar_chunks of decomposed_scalars) {
        const ell_sm = await create_ell_gpu(
            device,
            points,
            points_with_mont_coords,
            scalar_chunks,
            points_storage_buffer,
            num_rows,
            params,
        )
        ell_sms_serial.push(ell_sm)
    }

    return ell_sms
}

const fieldMath = new FieldMath()

export async function create_ell_gpu_x(
    device: GPUDevice,
    points: ExtPointType[],
    points_with_mont_coords: BigIntPoint[],
    scalar_chunks: number[],
    points_storage_buffer: GPUBuffer,
    num_rows: number,
    params: any,
) {
    const num_words = params.num_words
    const n0 = params.n0
    const rinv = params.rinv

    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    //let all_new_point_indices: number[] = []
    //const all_cluster_start_indices: number[] = []
    //const num_new_points_per_row: number[] = []
    ////const all_row_last_end_idx: number[] = []
    //let all_row_last_end_idx: number[] = []

    //let expected_new_points: any[] = []
    //let expected_new_scalar_chunks: number[] = []

    // TODO: don't pass zero chunks to the GPU

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices, row_last_end_idx } = prep_for_cluster_method(
            scalar_chunks,
            row_idx,
            num_rows,
        )
        console.log('row_idx:', row_idx)
        console.log('new_point_indices:', new_point_indices)
        console.log('cluster_start_indices:', cluster_start_indices)
        console.log('row_last_end_idx:', row_last_end_idx)
    }


    return new ELLSparseMatrix(data, col_idx, row_length)
}

// Create an ELL sparse matrix from all the points of the MSM and a set of
// scalar chunks
export async function create_ell_gpu(
    device: GPUDevice,
    points: ExtPointType[],
    points_with_mont_coords: BigIntPoint[],
    scalar_chunks: number[],
    points_storage_buffer: GPUBuffer,
    num_rows: number,
    params: any,
) {
    const num_words = params.num_words
    const n0 = params.n0
    const rinv = params.rinv

    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    let all_new_point_indices: number[] = []
    const all_cluster_start_indices: number[] = []
    const num_new_points_per_row: number[] = []
    //const all_row_last_end_idx: number[] = []
    const all_row_last_end_idx: number[] = []

    let expected_new_points: any[] = []
    let expected_new_scalar_chunks: number[] = []

    // TODO: don't pass zero chunks to the GPU

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices, row_last_end_idx } = prep_for_cluster_method(
            scalar_chunks,
            row_idx,
            num_rows,
        )

        const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
            points, 
            scalar_chunks,
            new_point_indices,
            cluster_start_indices,
        )

        expected_new_scalar_chunks = expected_new_scalar_chunks.concat(new_scalar_chunks)
        expected_new_points = expected_new_points.concat(new_points.map((x) => x.toAffine()))

        // Append data to the arrays which will be written to the GPU
        all_new_point_indices = all_new_point_indices.concat(new_point_indices)

        for (let i = 0; i < cluster_start_indices.length; i ++) {
            all_cluster_start_indices.push(cluster_start_indices[i] + (row_idx * (points.length / num_rows)))
        }

        num_new_points_per_row.push(all_cluster_start_indices.length)

        all_row_last_end_idx.push(row_last_end_idx)

        console.log('row_idx:', row_idx)
        console.log('new_point_indices:', new_point_indices)
        console.log('cluster_start_indices:', cluster_start_indices)
        console.log('row_last_end_idx:', row_last_end_idx)
        console.log('num_new_points_per_row:', num_new_points_per_row)
        console.log(all_row_last_end_idx)

        if (row_idx === 3) {
            //debugger
        }
    }

    // Convert inputs to bytes
    const scalar_chunks_bytes = numbers_to_u8s_for_gpu(scalar_chunks)
    const new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices)
    const cluster_start_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_start_indices)
    const num_new_points_per_row_bytes = numbers_to_u8s_for_gpu(num_new_points_per_row)
    const row_last_end_idx_bytes = numbers_to_u8s_for_gpu(all_row_last_end_idx)

    // Each x workgroup can operate on workgroup_size values at a time.
    const num_x_workgroups = 256

    const scalar_chunks_storage_buffer = device.createBuffer({
        size: scalar_chunks_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(scalar_chunks_storage_buffer, 0, scalar_chunks_bytes);

    const num_new_points_per_row_storage_buffer = device.createBuffer({
        size: num_new_points_per_row_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(num_new_points_per_row_storage_buffer, 0, num_new_points_per_row_bytes);

    const row_last_end_idx_storage_buffer = device.createBuffer({
        size: row_last_end_idx_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(num_new_points_per_row_storage_buffer, 0, num_new_points_per_row_bytes);

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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

    const shaderModule = device.createShaderModule({ code: shaderCode })

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    const start = Date.now()
    const new_point_indices_storage_buffer = device.createBuffer({
        size: new_point_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const cluster_start_indices_storage_buffer = device.createBuffer({
        size: cluster_start_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    device.queue.writeBuffer(new_point_indices_storage_buffer, 0, new_point_indices_bytes);
    device.queue.writeBuffer(cluster_start_indices_storage_buffer, 0, cluster_start_indices_bytes);

    const num_new_points = expected_new_points.length

    // Output buffers
    const new_points_storage_buffer = device.createBuffer({
        size: num_new_points * 320,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const new_scalar_chunks_storage_buffer = device.createBuffer({
        size: cluster_start_indices_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })

    const bindGroup = create_bind_group(
        device, 
        bindGroupLayout,
        [
            points_storage_buffer,
            scalar_chunks_storage_buffer,
            new_point_indices_storage_buffer,
            cluster_start_indices_storage_buffer,
            num_new_points_per_row_storage_buffer,
            row_last_end_idx_storage_buffer,
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
    const new_points_non_mont: any[] = []
    for (const pt of new_points) {
        const non = {
            x: fieldMath.Fp.mul(pt.x, rinv),
            y: fieldMath.Fp.mul(pt.y, rinv),
            t: fieldMath.Fp.mul(pt.t, rinv),
            z: fieldMath.Fp.mul(pt.z, rinv),
        }
        new_points_non_mont.push(bigIntPointToExtPointType(non, fieldMath).toAffine())
    }

    assert(new_points_non_mont.length === expected_new_points.length)

    for (let i = 0; i < new_points_non_mont.length; i ++) {
        if (new_points_non_mont[i].x !== expected_new_points[i].x) {
            console.log(`${i} / ${new_points_non_mont.length}`)
            debugger
            break
        }
        assert(new_points_non_mont[i].x === expected_new_points[i].x)
        assert(new_points_non_mont[i].y === expected_new_points[i].y)
    }

    data.push(new_points_non_mont)

    return new ELLSparseMatrix(data, col_idx, row_length)
}
