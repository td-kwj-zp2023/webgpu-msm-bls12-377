import assert from 'assert'
import {
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    execute_pipeline,
} from './gpu'
import { numbers_to_u8s_for_gpu } from './utils'

export const shader_invocation = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    shaderCode: string,
    x_y_coords_sb: GPUBuffer,
    t_z_coords_sb: GPUBuffer,
    out_x_y_sb: GPUBuffer,
    out_t_z_sb: GPUBuffer,
    num_points: number,
    num_words: number,
) => {
    assert(num_points <= 2 ** 16)

    const num_points_bytes = numbers_to_u8s_for_gpu([num_points])
    const num_points_sb = create_and_write_sb(device, num_points_bytes)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [ x_y_coords_sb, t_z_coords_sb, num_points_sb, out_x_y_sb, out_t_z_sb ],
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    const num_x_workgroups = 256
    const num_y_workgroups = 256

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, 1);

    const size = Math.ceil(num_points / 2) * 4 * num_words * 2
    commandEncoder.copyBufferToBuffer(
        out_x_y_sb,
        0,
        x_y_coords_sb,
        0,
        size,
    )
    commandEncoder.copyBufferToBuffer(
        out_t_z_sb,
        0,
        t_z_coords_sb,
        0,
        size,
    )

    return { out_x_y_sb, out_t_z_sb, num_points_sb }
}
