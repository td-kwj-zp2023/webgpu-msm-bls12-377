import assert from 'assert'
import {
    create_and_write_ub,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    execute_pipeline,
} from '../implementation/gpu'
import { numbers_to_u8s_for_gpu } from '../implementation/utils'

export const shader_invocation = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    shaderCode: string,
    x_coords_sb: GPUBuffer,
    y_coords_sb: GPUBuffer,
    t_coords_sb: GPUBuffer,
    z_coords_sb: GPUBuffer,
    out_x_sb: GPUBuffer,
    out_y_sb: GPUBuffer,
    out_t_sb: GPUBuffer,
    out_z_sb: GPUBuffer,
    num_points: number,
    num_words: number,
    workgroup_size: number,
) => {
    let num_z_workgroups = 2
    const compute_ideal_num_workgroups = (num_points: number) => {

        if (num_points <= num_z_workgroups) {
            return { num_x_workgroups: 1, num_y_workgroups: 1 }
        }

//         const m = Math.ceil(Math.log2(Math.sqrt(num_points / workgroup_size)))
//         // console.log("m is: ", m)
//         let num_x_workgroups = 2 ** m

//         const num_y_workgroups = (Math.ceil(num_points / num_x_workgroups / workgroup_size))

//         if (num_x_workgroups * num_y_workgroups * workgroup_size == num_points) {
//             num_x_workgroups = num_x_workgroups / 2
//         }

//         // console.log("num_x_workgroups is: ", num_x_workgroups)
//         // console.log("num_y_workgroups is: ", num_y_workgroups)
        
//         return { num_x_workgroups, num_y_workgroups }
//     }
    
//     const num_z_workgroups = 2
//     const compute_ideal_num_workgroups = (num_points: number) => {

        const m = Math.ceil(Math.log2(Math.sqrt(num_points / num_z_workgroups)))
        let num_x_workgroups = 2 ** m 
        let num_y_workgroups = 2 ** m

        if (num_x_workgroups * num_y_workgroups == (num_points / 2)) {
            num_z_workgroups = 1;
        }

        if (num_x_workgroups * num_y_workgroups == num_points) {
            num_x_workgroups = 2 ** (m - 1)
            num_y_workgroups = 2 ** (m - 1)
            num_z_workgroups = (num_points / 2) / (num_x_workgroups * num_y_workgroups);
        }

        return { num_x_workgroups, num_y_workgroups }
    }

    const { num_x_workgroups, num_y_workgroups } = compute_ideal_num_workgroups(num_points)

    const params_bytes = numbers_to_u8s_for_gpu(
        [
            num_y_workgroups,
            num_z_workgroups * 16,
        ],
    )
    const params_ub = create_and_write_ub(device, params_bytes)

    const bindGroupLayout = create_bind_group_layout(
        device,
        [
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'read-only-storage',
            'storage',
            'storage',
            'storage',
            'storage',
            'uniform',
        ],
    )
    const bindGroup = create_bind_group(
        device,
        bindGroupLayout,
        [
            x_coords_sb,
            y_coords_sb,
            t_coords_sb,
            z_coords_sb,
            out_x_sb,
            out_y_sb,
            out_t_sb,
            out_z_sb,
            params_ub,
        ]
    )

    const computePipeline = await create_compute_pipeline(
        device,
        [bindGroupLayout],
        shaderCode,
        'main',
    )

    execute_pipeline(commandEncoder, computePipeline, bindGroup, num_x_workgroups, num_y_workgroups, num_z_workgroups * 16)

    const size = (Math.ceil(num_points / 2) * 4 * num_words) * 16
    commandEncoder.copyBufferToBuffer(out_x_sb, 0, x_coords_sb, 0, size)
    commandEncoder.copyBufferToBuffer(out_y_sb, 0, y_coords_sb, 0, size)
    commandEncoder.copyBufferToBuffer(out_t_sb, 0, t_coords_sb, 0, size)
    commandEncoder.copyBufferToBuffer(out_z_sb, 0, z_coords_sb, 0, size)

    return {
        out_x_sb,
        out_y_sb,
        out_t_sb,
        out_z_sb,
        params_ub,
    }
}
