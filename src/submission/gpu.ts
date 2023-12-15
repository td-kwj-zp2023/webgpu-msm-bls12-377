import { BigIntPoint } from "../reference/types"

// Request GPU device
export const get_device = async (): Promise<GPUDevice> => {
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

// Create and write buffered memory accessible by the GPU memory space
export const create_and_write_sb = (
    device: GPUDevice,
    buffer: Uint8Array,
): GPUBuffer => {
    const sb = device.createBuffer({
        size: buffer.length,
        usage: read_write_buffer_usage
    })
    device.queue.writeBuffer(sb, 0, buffer)
    return sb
}

// Create buffered memory accessible by the GPU memory space
export const create_sb = (
    device: GPUDevice,
    size: number,
) => {
    return device.createBuffer({
        size,
        usage: read_write_buffer_usage
    })
}

// 
export const read_from_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    storage_buffers: GPUBuffer[],
) => {
    const staging_buffers: GPUBuffer[] = []
    for (const storage_buffer of storage_buffers) {
        const staging_buffer = device.createBuffer({
            size: storage_buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        })
        commandEncoder.copyBufferToBuffer(
            storage_buffer,
            0,
            staging_buffer,
            0,
            storage_buffer.size
        )
        staging_buffers.push(staging_buffer)
    }

    // Finish encoding commands and submit to GPU device command queue
    device.queue.submit([commandEncoder.finish()]);

    // Map staging buffers to read results back to JS
    const data = []
    for (let i = 0; i < staging_buffers.length; i ++) {
        await staging_buffers[i].mapAsync(
            GPUMapMode.READ,
            0,
            storage_buffers[i].size
        )
        const result_data = staging_buffers[i].getMappedRange(0, staging_buffers[i].size).slice(0)
        staging_buffers[i].unmap()
        data.push(new Uint8Array(result_data))
    }
    return data
}

// Bind Group Layout defines the input/output interface expected by the shader 
export const create_bind_group_layout = (
    device: GPUDevice,
    types: string[],
) => {
    const entries: any[] = []
    for (let i = 0; i < types.length; i ++) {
        entries.push({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: types[i] },
        })
    }
    return device.createBindGroupLayout({ entries })
}

// Bind groups maps buffers to data 
export const create_bind_group = (device: GPUDevice, layout: GPUBindGroupLayout, buffers: GPUBuffer[]) => {
    const entries: any[] = []
    for (let i = 0; i < buffers.length; i ++) {
        entries.push({
            binding: i,
            resource: { buffer: buffers[i] }
        })
    }
    return device.createBindGroup({ layout, entries })
}

// Asynchronously create pipeline
export const create_compute_pipeline = async (
    device: GPUDevice,
    bindGroupLayouts: GPUBindGroupLayout[],
    code: string,
    entryPoint: string,
) => {
    const m = device.createShaderModule({ code })

    // It's recommended to use createComputePipelineAsync instead of
    // createComputePipeline to prevent stalls:
    // https://www.khronos.org/assets/uploads/developers/presentations/WebGPU_Best_Practices_Google.pdf
    return device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts }),
        compute: { module: m, entryPoint }
    })
}

// Encode pipeline commands 
export const execute_pipeline = async (
    commandEncoder: GPUCommandEncoder,
    computePipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    num_x_workgroups: number,
    num_y_workgroups = 1,
    num_z_workgroups = 1,
) => {
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(num_x_workgroups, num_y_workgroups, num_z_workgroups)
    passEncoder.end()
}
 
export const read_write_buffer_usage = 
    GPUBufferUsage.STORAGE |
    GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST
