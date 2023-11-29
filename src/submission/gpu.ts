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

export const create_and_write_sb = (
    device: GPUDevice,
    buf: Uint8Array,
): GPUBuffer => {
    const sb = device.createBuffer({
        size: buf.length,
        usage: read_write_buffer_usage
    })
    device.queue.writeBuffer(sb, 0, buf)
    return sb
}

export const create_sb = (
    device: GPUDevice,
    size: number,
) => {
    return device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    })
}

export const read_from_gpu = async (
    device: GPUDevice,
    commandEncoder: GPUCommandEncoder,
    storage_buffers: GPUBuffer[],
) => {
    const staging_buffers: GPUBuffer[] = []
    for (const storage_buffer of storage_buffers) {
        const stb = device.createBuffer({
            size: storage_buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        })
        commandEncoder.copyBufferToBuffer(
            storage_buffer,
            0,
            stb,
            0,
            storage_buffer.size
        )
        staging_buffers.push(stb)
    }

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

export const read_write_buffer_usage = 
    GPUBufferUsage.STORAGE |
    GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST
