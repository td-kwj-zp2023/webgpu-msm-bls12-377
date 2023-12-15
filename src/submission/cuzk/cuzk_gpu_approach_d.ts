import { BigIntPoint } from "../../reference/types"
import {
    get_device,
    create_and_write_sb,
    read_from_gpu,
} from '../gpu'
import {
    u8s_to_numbers,
    numbers_to_u8s_for_gpu,
} from '../utils'
import { all_precomputation } from './create_csr'

import {
    convert_point_coords_to_mont_gpu,
    decompose_scalars_gpu,
    pre_aggregation_stage_1_gpu,
    pre_aggregation_stage_2_gpu,
} from './cuzk_gpu'

const word_size = 13

/*
 * End-to-end implementation of the cuZK MSM algorithm using Approach D (see 
 * https://github.com/TalDerei/webgpu-msm/pull/39#issuecomment-1820003395
 */
export const cuzk_gpu_approach_d = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const input_size = scalars.length
    const num_subtasks = 16

    // Each pass must use the same GPUDevice and GPUCommandEncoder, or else
    // storage buffers can't be reused across compute passes
    let device = await get_device()
    let commandEncoder = device.createCommandEncoder()

    // Convert the affine points to Montgomery form in the GPU
    let { point_x_y_sb, point_t_z_sb } =
        await convert_point_coords_to_mont_gpu(
            device,
            commandEncoder,
            baseAffinePoints,
            num_subtasks, 
            word_size,
            false,
        )

    // Decompose the scalars
    let scalar_chunks_sb = await decompose_scalars_gpu(
        device,
        commandEncoder,
        scalars,
        num_subtasks,
        word_size,
        false
    )

    // Read the scalar chunks from scalar_chunks_sb
    const [
        scalar_chunks_data,
        point_x_y_data,
        point_t_z_data,
    ]= await read_from_gpu(
        device,
        commandEncoder,
        [scalar_chunks_sb, point_x_y_sb, point_t_z_sb],
    )

    // Measure the amount of data read
    const total_gpu_to_cpu_data =
        scalar_chunks_data.length +
        point_x_y_data.length +
        point_t_z_data.length
    
    console.log(`Data transfer from GPU to CPU: ${total_gpu_to_cpu_data} bytes (${total_gpu_to_cpu_data / 1024 / 1024} MB)`)

    // The concatenation of the decomposed scalar chunks across all subtasks
    const all_computed_chunks = u8s_to_numbers(scalar_chunks_data)

    // Recreate the commandEncoder, since read_from_gpu ran finish() on it, as
    // well as the device, since storage buffers can't be shared across devices
    device = await get_device()
    commandEncoder = device.createCommandEncoder()

    // Recreate the storage buffers under the new device object
    scalar_chunks_sb = create_and_write_sb(device, scalar_chunks_data)
    point_x_y_sb = create_and_write_sb(device, point_x_y_data)
    point_t_z_sb = create_and_write_sb(device, point_t_z_data)

    let total_cpu_to_gpu_data =
        scalar_chunks_data.length +
        point_x_y_data.length +
        point_t_z_data.length

    let total_precomputation_ms = 0
    const num_rows = 16
    for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
        const start = Date.now()
        const scalar_chunks = all_computed_chunks.slice(
            subtask_idx * input_size,
            subtask_idx * input_size + input_size,
        )

        // Precomputation in CPU
        const {
            all_new_point_indices,
            all_cluster_start_indices,
            all_cluster_end_indices,
            all_single_point_indices,
            all_single_scalar_chunks,
            row_ptr,
        } = all_precomputation(scalar_chunks, num_rows)

        const all_new_point_indices_bytes = numbers_to_u8s_for_gpu(all_new_point_indices)
        const all_cluster_start_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_start_indices)
        const all_cluster_end_indices_bytes = numbers_to_u8s_for_gpu(all_cluster_end_indices)

        const new_point_indices_sb = create_and_write_sb(device, all_new_point_indices_bytes)
        const cluster_start_indices_sb = create_and_write_sb(device, all_cluster_start_indices_bytes)
        const cluster_end_indices_sb = create_and_write_sb(device, all_cluster_end_indices_bytes)
        const elapsed = Date.now() - start
        total_precomputation_ms += elapsed

        total_cpu_to_gpu_data +=
            all_new_point_indices_bytes.length +
            all_cluster_start_indices_bytes.length +
            all_cluster_end_indices_bytes.length

        // TOOD: figure out where these go:
        // - all_single_point_indices
        // - all_single_scalar_chunks
        // - row_ptr

        const {
            new_point_x_y_sb,
            new_point_t_z_sb,
        } = await pre_aggregation_stage_1_gpu(
            device,
            commandEncoder,
            input_size,
            point_x_y_sb,
            point_t_z_sb,
            new_point_indices_sb,
            cluster_start_indices_sb,
            cluster_end_indices_sb,
            false,
        )

        const new_scalar_chunks_sb = await pre_aggregation_stage_2_gpu(
            device,
            commandEncoder,
            input_size,
            scalar_chunks_sb,
            cluster_start_indices_sb,
            new_point_indices_sb,
            false,
        )
    }
    console.log(`all_precomputation for ${num_subtasks} subtasks took: ${total_precomputation_ms}ms`)
    console.log(`Data transfer from CPU to GPU: ${total_cpu_to_gpu_data} bytes (${total_cpu_to_gpu_data / 1024 / 1024} MB)`)

    return { x: BigInt(1), y: BigInt(0) }
}


