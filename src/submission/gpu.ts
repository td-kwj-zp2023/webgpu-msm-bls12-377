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

export const create_bind_group = (device: GPUDevice,layout: GPUBindGroupLayout, buffers: GPUBuffer[]) => {
    const entries: any[] = []
    for (let i = 0; i < buffers.length; i ++) {
        entries.push({
            binding: i,
            resource: { buffer: buffers[i] }
        })
    }
    return device.createBindGroup({ layout, entries })
}
