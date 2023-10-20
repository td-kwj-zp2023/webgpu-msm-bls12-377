import { BigIntPoint } from "../../reference/types";
import { DenseMatrix, ELLSparseMatrix, CSRSparseMatrix } from '../matrices/matrices'; 
import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { strict as assert } from 'assert';
import { bigIntToU32Array, generateRandomFields } from '../../reference/webgpu/utils';
import { Field } from "@noble/curves/abstract/modular";
import { points_to_u8s_for_gpu } from '../../submission/utils'
import mustache from 'mustache'
import shader from '../wgsl/mont_pro_optimised.template.wgsl'

/**
 * Top-Level Overview
 *  1. Buffers — input and output data
 *  2. Shaders — Specified in WGSL, execute a computing instruction
 *  3. Binding groups and layouts — mapping between buffers and shaders
 */
export async function execute_cuzk_wgsl(
    inputSize: number,
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<any> {    
    // Define number of workgroups
    const num_x_workgroups = 1

    // λ-bit scalars
    const lambda = 256

    // s-bit window size 
    const s = 16

    /**
     * 1. Intialize WebGPU 
     */

    // Returns javascript promise that asynchronously resolves with GPU adapter
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
    });
    if (!adapter) { console.log("Failed to get WebGPU adapter!"); return; }

    // Returns a promise that resolves with a GPU device. 
    const device = await adapter.requestDevice();

    /** 
     * 2. Create Buffered Memory Accessible by the GPU Memory Space 
     */

    const input_bytes = points_to_u8s_for_gpu(baseAffinePoints, Math.ceil(lambda / s), s)

    // const input_bytes = bigints_to_u8_for_gpu(inputs, Math.ceil(lambda / s), s)

    // Points buffer 
    const points_gpu = device.createBuffer({
        mappedAtCreation: true,
        size: input_bytes.length,    
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // GPU buffer method that retrieves the raw binary data buffer 
    const pointsBuffer = points_gpu.getMappedRange();
    new Int8Array(pointsBuffer).set(input_bytes);

    // Enables the GPU to take control -- and prevents race conditions where GPU/CPU access memory at the same time
    points_gpu.unmap();

    // Scalar buffer 
    const scalars_gpu = device.createBuffer({
        mappedAtCreation: true,
        size: input_bytes.length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const scalarsBuffer = scalars_gpu.getMappedRange();
    new Int8Array(scalarsBuffer).set(input_bytes);
    scalars_gpu.unmap();

    // Staging buffer 
    const stagingBuffer = device.createBuffer({
        size: input_bytes.length,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    /** 
     * 3. Define Bind Group Layouts and Bind Groups 
     */

    // Bind Group Layout defines the input/output interface expected by the shader 
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ]
    });

    // Bind Group represents the actual input/output data for the shader, and associate with GPU buffers 
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: points_gpu } },
            { binding: 1, resource: { buffer: scalars_gpu } },
            { binding: 2, resource: { buffer: stagingBuffer } },
        ]
    });

    /**
     * 4. Load Compute Shaders (WGSL)
     */

    const shaderCode = mustache.render(
        shader,
        {}
    )
    const shaderModule = device.createShaderModule({code: shaderCode})

    /**
     * 5. Setup Compute Pipeline 
     */ 

    /** device.createComputePipeline() creates pipeline with bind group layout and compute stage as arguments */
    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute : {
            module: shaderModule,   
            entryPoint: "main"     

        }
    });

    /** 
     * 6. Execute compute pass 
     */

    // Returns a Javascript object that encodes a batch if "buffered" GPU commands 
    const commandEncoder = device.createCommandEncoder();

    // Start timer
    const start = Date.now()

    // Set pipeline 
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);

    // Set bind group at index 0 (corresponding with group(0) in WGSL) 
    passEncoder.setBindGroup(0, bindGroup);

    // Set the number of workgroups dispatched for the execution of a kernel function 
    passEncoder.dispatchWorkgroups(
        Math.ceil(num_x_workgroups),
        Math.ceil(num_x_workgroups)
    );

    // Ends the compute pass encoder 
    passEncoder.end();

    /**
     * 7. Copy Buffer 
     */
    
    // Add the command to GPU device command queue for later execution 
    commandEncoder.copyBufferToBuffer(
        points_gpu,                                 
        0,                                                 
        stagingBuffer,                      
        0,                                      
        input_bytes.length,                   
    );

    /** 
     * 8. Execute command queue
     */
    
    // Finish encoding commands and submit to GPU device command queue */
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    // Map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, 
        input_bytes.length
    );
    const copyArrayBuffer = stagingBuffer.getMappedRange(0, input_bytes.length)
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();
    const dataBuf = new Uint32Array(data);

    const elapsed = Date.now() - start
}
