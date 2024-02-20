/**
 * Copyright 2024 Tal Derei and Koh Wei Jie. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Koh Wei Jie and Tal Derei
 */

import assert from "assert";
import { readBigIntsFromBufferLE } from '../reference/webgpu/utils';
import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { ShaderManager } from "./implementation/cuzk/shader_manager";
import {
  get_device,
  create_and_write_sb,
  create_and_write_ub,
  create_bind_group,
  create_bind_group_layout,
  create_compute_pipeline,
  create_sb,
  read_from_gpu,
  execute_pipeline,
} from "./implementation/cuzk/gpu";
import {
  u8s_to_bigints,
  format_points_buffer_for_gpu,
  u8s_to_numbers,
  u8s_to_numbers_32,
  from_words_le,
  format_buffer_for_gpu,
  numbers_to_u8s_for_gpu,
  compute_misc_params,
  decompose_scalars_signed,
} from "./implementation/cuzk/utils";
import { cpu_transpose } from "./implementation/cuzk/transpose";
import { cpu_smvp_signed } from "./implementation/cuzk/smvp";
import {
  parallel_bucket_reduction_1,
  parallel_bucket_reduction_2,
} from './implementation/cuzk/bpr'

const p = BigInt(
  "8444461749428370424248824938781546531375899335154063827935233455917409239041",
);
const word_size = 13;
const params = compute_misc_params(p, word_size);
const num_words = params.num_words;
const r = params.r;
const rinv = params.rinv;

import { FieldMath } from "../reference/utils/FieldMath";
const fieldMath = new FieldMath();

/*
 * End-to-end implementation of the modified cuZK MSM algorithm by Lu et al,
 * 2022: https://eprint.iacr.org/2022/1321.pdf
 * Many aspects of cuZK were adapted and modified for our submission, and some
 * aspects were omitted. As such, please refer to the documentation we have
 * written for a more accurate description of our work. We also used techniques
 * by previous ZPrize contestations. In summary, we took the following
 * approach:
 * 1. Perform as much of the computation within the GPU as possible, in order
 *    to minimse CPU-GPU and GPU-CPU data transfer, which is slow.
 * 2. Use optimisations inspired by previous years' submissions, such as:
 *    - Montgomery multiplication with smaller limb sizes
 *    - Signed bucket indices
 * 3. Careful memory management to stay within WebGPU's default buffer size
 *    limits.
 * 4. Perform the final computation of the MSM result from the subtask results
 *    (Horner's rule) in the CPU instead of the GPU, as the number of points is
 *    small, and the time taken to compile a shader to perform this computation
 *    is greater than the time it takes for the CPU to do so.
 */
export const compute_msm = async (
  bufferPoints: BigIntPoint[] | U32ArrayPoint[] | Buffer,
  bufferScalars: bigint[] | Uint32Array[] | Buffer,
  log_result = true,
  force_recompile = false,
): Promise<{ x: bigint; y: bigint }> => {
  const input_size = bufferScalars.length / 32;

  const chunk_size = input_size >= 65536 ? 16 : 4;

  const shaderManager = new ShaderManager(
    word_size,
    chunk_size,
    input_size,
    force_recompile,
  );

  const num_columns = 2 ** chunk_size;
  const num_rows = Math.ceil(input_size / num_columns);

  const num_chunks_per_scalar = Math.ceil(256 / chunk_size);
  const num_subtasks = num_chunks_per_scalar;

  // Each pass must use the same GPUDevice and GPUCommandEncoder, or else
  // storage buffers can't be reused across compute passes
  const device = await get_device();

  // Convert the affine points to Montgomery form and decompose the scalars
  // using a single shader

  let c_workgroup_size = 64;
  let c_num_x_workgroups = 128;
  let c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;

  if (input_size <= 256) {
    c_workgroup_size = input_size;
    c_num_x_workgroups = 1;
    c_num_y_workgroups = 1;
  } else if (input_size > 256 && input_size <= 32768) {
    c_workgroup_size = 64;
    c_num_x_workgroups = 4;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 32768 && input_size <= 65536) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 8;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 65536 && input_size <= 131072) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 8;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 131072 && input_size <= 262144) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 262144 && input_size <= 524288) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  } else if (input_size > 524288 && input_size <= 1048576) {
    c_workgroup_size = 256;
    c_num_x_workgroups = 32;
    c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
  }

  const c_shader = shaderManager.gen_convert_points_and_decomp_scalars_shader(
    c_workgroup_size,
    c_num_y_workgroups,
    num_subtasks,
    num_columns,
  );

  // Create single command encoder for device
  const commandEncoder = device.createCommandEncoder();

  const { point_x_sb, point_y_sb, scalar_chunks_sb } =
    await convert_point_coords_and_decompose_shaders(
      c_shader,
      c_num_x_workgroups,
      c_num_y_workgroups,
      device,
      commandEncoder,
      bufferPoints as Buffer,
      num_words,
      word_size,
      bufferScalars as Buffer,
      num_subtasks,
      num_columns,
      chunk_size,
    );

  // Buffers to  store the SMVP result (the bucket sum). They are overwritten
  // per iteration
  const bucket_sum_coord_bytelength =
    (num_columns / 2) * num_words * 4 * num_subtasks;
  const bucket_sum_x_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_y_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_t_sb = create_sb(device, bucket_sum_coord_bytelength);
  const bucket_sum_z_sb = create_sb(device, bucket_sum_coord_bytelength);

  const t_shader = shaderManager.gen_transpose_shader(num_subtasks);

  // Transpose
  const { all_csc_col_ptr_sb, all_csc_val_idxs_sb } = await transpose_gpu(
    t_shader,
    device,
    commandEncoder,
    input_size,
    num_columns,
    num_rows,
    num_subtasks,
    scalar_chunks_sb,
  );

  const half_num_columns = num_columns / 2;
  let s_workgroup_size = 256;
  let s_num_x_workgroups = 64;
  let s_num_y_workgroups =
    half_num_columns / s_workgroup_size / s_num_x_workgroups;
  let s_num_z_workgroups = num_subtasks;

  if (half_num_columns < 32768) {
    s_workgroup_size = 32;
    s_num_x_workgroups = 1;
    s_num_y_workgroups = Math.ceil(
      half_num_columns / s_workgroup_size / s_num_x_workgroups,
    );
  }

  if (num_columns < 256) {
    s_workgroup_size = 1;
    s_num_x_workgroups = half_num_columns;
    s_num_y_workgroups = 1;
    s_num_z_workgroups = 1;
  }

  // This is a dynamic variable that determines the number
  // of CSR matrices processed per invocation of the shader. 
  // TODO: find a safe default
  const num_subtask_chunk_size = 1

  const smvp_shader = shaderManager.gen_smvp_shader(
    s_workgroup_size,
    num_columns,
  );

  // SMVP and multiplication by the bucket index
  //const offset_debug = 0
  //const should_debug = true
  for (let offset = 0; offset < num_subtasks; offset += num_subtask_chunk_size) {
    await smvp_gpu(
      smvp_shader,
      s_num_x_workgroups / (num_subtasks / num_subtask_chunk_size),
      s_num_y_workgroups,
      s_num_z_workgroups,
      offset,
      num_subtask_chunk_size,
      device,
      commandEncoder,
      num_subtasks,
      num_columns,
      input_size,
      chunk_size,
      all_csc_col_ptr_sb,
      point_x_sb,
      point_y_sb,
      all_csc_val_idxs_sb,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      //should_debug && offset === offset_debug,
    )
    //if (should_debug && offset === offset_debug) { device.destroy(); return { x: BigInt(0), y: BigInt(1) } }
  }

  // There will be 2 ** 16 buckets
  const b_num_x_workgroups = 1
  const b_workgroup_size = 256
  const b_num_threads = b_num_x_workgroups * b_workgroup_size

  // Output of the parallel bucket points reduction (BPR) shader
  const g_points_coord_bytelength = num_subtasks * b_num_threads * num_words * 4;
  const g_points_x_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_y_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_t_sb = create_sb(device, g_points_coord_bytelength);
  const g_points_z_sb = create_sb(device, g_points_coord_bytelength);

  const bpr_shader = shaderManager.gen_bpr_shader(b_workgroup_size) 

  //const b_1_debug = 0
  for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
    // Bucket points reduction (BPR) - stage 1
    await bpr_1(
      bpr_shader,
      subtask_idx,
      num_subtasks,
      b_num_x_workgroups,
      b_workgroup_size,
      num_columns,
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
      //subtask_idx === b_1_debug,
    )
    //if (subtask_idx === b_1_debug) { device.destroy(); return { x: BigInt(0), y: BigInt(1) } }
  }

  //const b_2_debug = 0
  for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx ++) {
    // Bucket points reduction (BPR) - stage 2
    await bpr_2(
      bpr_shader,
      subtask_idx,
      num_subtasks,
      b_num_x_workgroups,
      b_workgroup_size,
      num_columns,
      device,
      commandEncoder,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
      //subtask_idx === b_2_debug,
    )
    //if (subtask_idx === b_2_debug) { device.destroy(); return { x: BigInt(0), y: BigInt(1) } }
  }

  const data = await read_from_gpu(device, commandEncoder, [
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
  ]);

  // Destroy the GPU device object
  device.destroy();

  const g_points_x_mont_coords = u8s_to_bigints(data[0], num_words, word_size);
  const g_points_y_mont_coords = u8s_to_bigints(data[1], num_words, word_size);
  const g_points_t_mont_coords = u8s_to_bigints(data[2], num_words, word_size);
  const g_points_z_mont_coords = u8s_to_bigints(data[3], num_words, word_size);

  const points: ExtPointType[] = [];

  // For a small number of points, this is extremely fast in the CPU
  for (let i = 0; i < num_subtasks; i ++) {
    let point = fieldMath.customEdwards.ExtendedPoint.ZERO;
    for (let j = 0; j < b_num_threads; j ++) {
      const reduced_point = fieldMath.createPoint(
        fieldMath.Fp.mul(g_points_x_mont_coords[i * b_num_threads + j], rinv),
        fieldMath.Fp.mul(g_points_y_mont_coords[i * b_num_threads + j], rinv),
        fieldMath.Fp.mul(g_points_t_mont_coords[i * b_num_threads + j], rinv),
        fieldMath.Fp.mul(g_points_z_mont_coords[i * b_num_threads + j], rinv),
      );
      //if (!reduced_point.equals(zero)) { reduced_point.assertValidity(); }
      point = point.add(reduced_point);
      //if (i == 15) { debugger }
    }
    points.push(point);
  }

  // Calculate the final result using Horner's method (Formula 3 of the cuZK
  // paper)
  const m = BigInt(2) ** BigInt(chunk_size);
  // The last scalar chunk is the most significant digit (base m)
  let result = points[points.length - 1];
  for (let i = points.length - 2; i >= 0; i--) {
    result = result.multiply(m);
    result = result.add(points[i]);
  }

  if (log_result) {
    console.log(result.toAffine());
  }
  return result.toAffine();
  //return { x: BigInt(0), y: BigInt(1) }
};

/*
 * Convert the affine points to Montgomery form, and decompose scalars into
 * chunk_size windows using the signed bucket index technique.

 * ASSUMPTION: the vast majority of WebGPU-enabled consumer devices have a
 * maximum buffer size of at least 268435456 bytes.
 * 
 * The default maximum buffer size is 268435456 bytes. Since each point
 * consumes 320 bytes, a maximum of around 2 ** 19 points can be stored in a
 * single buffer. If, however, we use 2 buffers - one for each point coordinate
 * X and Y - we can support larger input sizes.
 * Our implementation, however, will only support up to 2 ** 20 points as that
 * is the maximum input size for the ZPrize competition.
 *
 * Furthremore, there is a limit of 8 storage buffers per shader. As such, we
 * do not calculate the T and Z coordinates in this shader. Rather, we do so in
 * the SMVP shader.
 * 
 * Note that The test harness readme at
 * https://github.com/demox-labs/webgpu-msm states: "The submission should
 * produce correct outputs on input vectors with length up to 2^20. The
 * evaluation will be using input randomly sampled from size 2^16 ~ 2^20."
*/
export const convert_point_coords_and_decompose_shaders = async (
  shaderCode: string,
  num_x_workgroups: number,
  num_y_workgroups: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bufferPoints: Buffer,
  num_words: number,
  word_size: number,
  scalars_buffer: Buffer,
  num_subtasks: number,
  num_columns: number,
  chunk_size: number,
  debug = false,
) => {
  assert(num_subtasks * chunk_size === 256);
  const input_size = scalars_buffer.length / 32

  const start = Date.now()
  const { x_coords_bytes, y_coords_bytes } = format_points_buffer_for_gpu(bufferPoints)

  // Convert scalars to bytes
  const scalars_bytes = format_buffer_for_gpu(scalars_buffer)
  const elapsed = Date.now() - start
  console.log('elapsed:', elapsed)

  // Input buffers
  const x_coords_sb = create_and_write_sb(device, x_coords_bytes);
  const y_coords_sb = create_and_write_sb(device, y_coords_bytes);
  const scalars_sb = create_and_write_sb(device, scalars_bytes);

  // Output buffers
  const point_x_sb = create_sb(device, input_size * num_words * 4);
  const point_y_sb = create_sb(device, input_size * num_words * 4);
  const scalar_chunks_sb = create_sb(device, input_size * num_subtasks * 4);

  // Uniform param buffer
  const params_bytes = numbers_to_u8s_for_gpu([input_size]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);
  const bindGroup = create_bind_group(device, bindGroupLayout, [
    x_coords_sb,
    y_coords_sb,
    scalars_sb,
    point_x_sb,
    point_y_sb,
    scalar_chunks_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    1,
  );

  // Debug the output of the shader. This should **not** be run in
  // production.
  if (debug) {
    const data = await read_from_gpu(device, commandEncoder, [
      point_x_sb,
      point_y_sb,
      scalar_chunks_sb,
    ]);

    // Verify point coords
    const computed_x_coords = u8s_to_bigints(data[0], num_words, word_size);
    const computed_y_coords = u8s_to_bigints(data[1], num_words, word_size);

    const x_coords: bigint[] = []
    const y_coords: bigint[] = []

    for (let i = 0; i < input_size; i ++) {
      const x_slice = bufferPoints.slice(i * 64, i * 64 + 32);
      x_coords.push(from_words_le(new Uint16Array(x_slice), 32, 8));
      const y_slice = bufferPoints.slice(i * 64, i * 64 + 32);
      y_coords.push(from_words_le(new Uint16Array(y_slice), 32, 8));
    }

    for (let i = 0; i < input_size; i++) {
      const expected_x = (x_coords[i] * r) % p;
      const expected_y = (y_coords[i] * r) % p;

      if (
        !(
          expected_x === computed_x_coords[i] &&
          expected_y === computed_y_coords[i]
        )
      ) {
        console.log("mismatch at", i);
        break;
      }
    }

    // Verify scalar chunks
    const computed_chunks = u8s_to_numbers(data[2]);

    const scalars = readBigIntsFromBufferLE(scalars_buffer)

    const expected = decompose_scalars_signed(
      scalars,
      num_subtasks,
      chunk_size,
    );

    for (let j = 0; j < expected.length; j++) {
      let z = 0;
      for (let i = j * input_size; i < (j + 1) * input_size; i++) {
        if (computed_chunks[i] !== expected[j][z]) {
          throw Error(`scalar decomp mismatch at ${i}`);
        }
        z++;
      }
    }
  }

  return { point_x_sb, point_y_sb, scalar_chunks_sb };
};

/*
 * Perform a modified version of CSR matrix transposition, which comes before
 * SMVP. Essentially, this step generates the point indices for each thread in
 * the SMVP step which corresponds to a particular bucket.
 */
export const transpose_gpu = async (
  shaderCode: string,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  input_size: number,
  num_columns: number,
  num_rows: number,
  num_subtasks: number,
  scalar_chunks_sb: GPUBuffer,
  debug = false,
): Promise<{
  all_csc_col_ptr_sb: GPUBuffer;
  all_csc_val_idxs_sb: GPUBuffer;
}> => {
  /*
   * n = number of columns (before transposition)
   * m = number of rows (before transposition)
   * nnz = number of nonzero elements
   *
   * Given:
   *   - csr_col_idx (nnz) (aka the new_scalar_chunks)
   *
   * Output the transpose of the above:
   *   - csc_col_ptr (m + 1)
   *      - The column index of each nonzero element
   *   - csc_val_idxs (nnz)
   *      - The new index of each nonzero element
   *
   * Not computed as it's not used:
   *   - csc_row_idx (nnz)
   *      - The cumulative sum of the number of nonzero elements per row
   */

  const all_csc_col_ptr_sb = create_sb(
    device,
    num_subtasks * (num_columns + 1) * 4,
  );
  const all_csc_val_idxs_sb = create_sb(device, scalar_chunks_sb.size);
  const all_curr_sb = create_sb(device, num_subtasks * num_columns * 4);

  const params_bytes = numbers_to_u8s_for_gpu([
    num_rows,
    num_columns,
    input_size,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    scalar_chunks_sb,
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    all_curr_sb,
    params_ub,
  ]);

  const num_x_workgroups = 1;
  const num_y_workgroups = 1;

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    1,
  );

  // Debug the output of the shader. This should **not** be run in
  // production.
  if (debug) {
    const data = await read_from_gpu(device, commandEncoder, [
      all_csc_col_ptr_sb,
      all_csc_val_idxs_sb,
      scalar_chunks_sb,
    ]);

    const all_csc_col_ptr_result = u8s_to_numbers_32(data[0]);
    const all_csc_val_idxs_result = u8s_to_numbers_32(data[1]);
    const new_scalar_chunks = u8s_to_numbers_32(data[2]);

    // Verify the output of the shader
    const expected = cpu_transpose(
      new_scalar_chunks,
      num_columns,
      num_rows,
      num_subtasks,
      input_size,
    );

    assert(
      expected.all_csc_col_ptr.toString() === all_csc_col_ptr_result.toString(),
      "all_csc_col_ptr mismatch",
    );
    assert(
      expected.all_csc_vals.toString() === all_csc_val_idxs_result.toString(),
      "all_csc_vals mismatch",
    );
  }

  return {
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
  };
};

/*
 * Compute the bucket sums and perform scalar multiplication with the bucket
 * indices.
 */
export const smvp_gpu = async (
  shaderCode: string,
  num_x_workgroups: number,
  num_y_workgroups: number,
  num_z_workgroups: number,
  offset: number,
  num_subtask_chunk_size: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  num_subtasks: number,
  num_csr_cols: number,
  input_size: number,
  chunk_size: number,
  all_csc_col_ptr_sb: GPUBuffer,
  point_x_sb: GPUBuffer,
  point_y_sb: GPUBuffer,
  all_csc_val_idxs_sb: GPUBuffer,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  debug = false,
) => {
  const params_bytes = numbers_to_u8s_for_gpu([
    input_size,
    num_y_workgroups,
    num_z_workgroups,
    offset
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    all_csc_col_ptr_sb,
    all_csc_val_idxs_sb,
    point_x_sb,
    point_y_sb,
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main",
  );

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  // Debug the output of the shader. This should **not** be run in production.
  if (debug) {
    const data = await read_from_gpu(device, commandEncoder, [
      all_csc_col_ptr_sb,
      all_csc_val_idxs_sb,
      point_x_sb,
      point_y_sb,
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
    ]);

    const all_csc_col_ptr_sb_result = u8s_to_numbers_32(data[0]);
    const all_csc_val_idxs_result = u8s_to_numbers_32(data[1]);
    const point_x_sb_result = u8s_to_bigints(data[2], num_words, word_size);
    const point_y_sb_result = u8s_to_bigints(data[3], num_words, word_size);
    const bucket_sum_x_sb_result = u8s_to_bigints(
      data[4],
      num_words,
      word_size,
    );
    const bucket_sum_y_sb_result = u8s_to_bigints(
      data[5],
      num_words,
      word_size,
    );
    const bucket_sum_t_sb_result = u8s_to_bigints(
      data[6],
      num_words,
      word_size,
    );
    const bucket_sum_z_sb_result = u8s_to_bigints(
      data[7],
      num_words,
      word_size,
    );

    // Note: assertion checks take a long time

    // Convert GPU output out of Montgomery coordinates
    const bigIntPointToExtPointType = (bip: BigIntPoint): ExtPointType => {
      return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
    };
    const output_points_gpu: ExtPointType[] = [];
    for (
      let i = offset * (num_csr_cols / 2);
      i < offset * (num_csr_cols / 2) + num_csr_cols / 2;
      i++
    ) {
      const non = {
        x: fieldMath.Fp.mul(bucket_sum_x_sb_result[i], rinv),
        y: fieldMath.Fp.mul(bucket_sum_y_sb_result[i], rinv),
        t: fieldMath.Fp.mul(bucket_sum_t_sb_result[i], rinv),
        z: fieldMath.Fp.mul(bucket_sum_z_sb_result[i], rinv),
      };
      output_points_gpu.push(bigIntPointToExtPointType(non));
    }

    const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const input_points: ExtPointType[] = [];
    for (let i = 0; i < input_size; i++) {
      const x = fieldMath.Fp.mul(point_x_sb_result[i], rinv);
      const y = fieldMath.Fp.mul(point_y_sb_result[i], rinv);
      const t = fieldMath.Fp.mul(x, y);
      const pt = fieldMath.createPoint(x, y, t, BigInt(1));
      if (!pt.equals(zero)) { pt.assertValidity(); }
      input_points.push(pt);
    }

    // Calculate SMVP in CPU
    const output_points_cpu: ExtPointType[] = cpu_smvp_signed(
      offset,
      input_size,
      num_csr_cols,
      chunk_size,
      all_csc_col_ptr_sb_result,
      all_csc_val_idxs_result,
      input_points,
      fieldMath,
    );

    // Transform results into affine representation
    const output_points_affine_cpu = output_points_cpu.map((x) =>
      x.toAffine(),
    );
    const output_points_affine_gpu = output_points_gpu.map((x) =>
      x.toAffine(),
    );

    // Assert CPU and GPU output
    for (let i = 0; i < output_points_affine_gpu.length; i++) {
      assert(
        output_points_affine_gpu[i].x === output_points_affine_cpu[i].x &&
        output_points_affine_gpu[i].y === output_points_affine_cpu[i].y,
        `failed at offset ${offset}, i = ${i}`,
      );
    }
  }

  return {
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
  };
};

const bpr_1 = async (
  shaderCode: string,
  subtask_idx: number,
  num_subtasks: number,
  num_x_workgroups: number,
  workgroup_size: number,
  num_columns: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  debug = false,
) => {
  let original_bucket_sum_x_sb;
  let original_bucket_sum_y_sb;
  let original_bucket_sum_t_sb;
  let original_bucket_sum_z_sb;

  // Debug the output of the shader. This should **not** be run in
  // production.
  if (debug) {
    original_bucket_sum_x_sb = create_sb(device, bucket_sum_x_sb.size);
    original_bucket_sum_y_sb = create_sb(device, bucket_sum_y_sb.size);
    original_bucket_sum_t_sb = create_sb(device, bucket_sum_t_sb.size);
    original_bucket_sum_z_sb = create_sb(device, bucket_sum_z_sb.size);

    commandEncoder.copyBufferToBuffer(
      bucket_sum_x_sb,
      0,
      original_bucket_sum_x_sb,
      0,
      bucket_sum_x_sb.size,
    );
    commandEncoder.copyBufferToBuffer(
      bucket_sum_y_sb,
      0,
      original_bucket_sum_y_sb,
      0,
      bucket_sum_y_sb.size,
    );
    commandEncoder.copyBufferToBuffer(
      bucket_sum_t_sb,
      0,
      original_bucket_sum_t_sb,
      0,
      bucket_sum_t_sb.size,
    );
    commandEncoder.copyBufferToBuffer(
      bucket_sum_z_sb,
      0,
      original_bucket_sum_z_sb,
      0,
      bucket_sum_z_sb.size,
    );
  }

  // Parameters as a uniform buffer
  const params_bytes = numbers_to_u8s_for_gpu([
    subtask_idx,
    num_columns,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main_1",
  );

  const num_threads = num_x_workgroups * workgroup_size
  const num_y_workgroups = 1
  const num_z_workgroups = 1
  //console.log({ num_threads, num_x_workgroups, workgroup_size })

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  if (
    debug &&
    original_bucket_sum_x_sb != undefined && // prevent TS warnings
    original_bucket_sum_y_sb != undefined &&
    original_bucket_sum_t_sb != undefined &&
    original_bucket_sum_z_sb != undefined
  ) {
    const data = await read_from_gpu(device, commandEncoder, [
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
      original_bucket_sum_x_sb,
      original_bucket_sum_y_sb,
      original_bucket_sum_t_sb,
      original_bucket_sum_z_sb,
    ])

    // The number of buckets per subtask
    const n = num_columns / 2

    const start = subtask_idx * n * num_words * 4
    const end = (subtask_idx * n + n) * num_words * 4

    const m_points_x_mont_coords = u8s_to_bigints(data[0].slice(start, end), num_words, word_size);
    const m_points_y_mont_coords = u8s_to_bigints(data[1].slice(start, end), num_words, word_size);
    const m_points_t_mont_coords = u8s_to_bigints(data[2].slice(start, end), num_words, word_size);
    const m_points_z_mont_coords = u8s_to_bigints(data[3].slice(start, end), num_words, word_size);

    const g_points_x_mont_coords = u8s_to_bigints(data[4], num_words, word_size);
    const g_points_y_mont_coords = u8s_to_bigints(data[5], num_words, word_size);
    const g_points_t_mont_coords = u8s_to_bigints(data[6], num_words, word_size);
    const g_points_z_mont_coords = u8s_to_bigints(data[7], num_words, word_size);

    const original_bucket_sum_x_mont_coords = u8s_to_bigints(data[8].slice(start, end), num_words, word_size);
    const original_bucket_sum_y_mont_coords = u8s_to_bigints(data[9].slice(start, end), num_words, word_size);
    const original_bucket_sum_t_mont_coords = u8s_to_bigints(data[10].slice(start, end), num_words, word_size);
    const original_bucket_sum_z_mont_coords = u8s_to_bigints(data[11].slice(start, end), num_words, word_size);

    const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

    // Convert the bucket sums out of Montgomery form
    const original_bucket_sums: ExtPointType[] = []
    for (let i = 0; i < n; i ++) {
      const pt = fieldMath.createPoint(
        fieldMath.Fp.mul(original_bucket_sum_x_mont_coords[i], rinv),
        fieldMath.Fp.mul(original_bucket_sum_y_mont_coords[i], rinv),
        fieldMath.Fp.mul(original_bucket_sum_t_mont_coords[i], rinv),
        fieldMath.Fp.mul(original_bucket_sum_z_mont_coords[i], rinv),
      );
      if (!pt.equals(zero)) { pt.assertValidity(); }
      original_bucket_sums.push(pt);
    }

    const m_points: ExtPointType[] = []
    for (let i = 0; i < n; i ++) {
      const pt = fieldMath.createPoint(
        fieldMath.Fp.mul(m_points_x_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_y_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_t_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_z_mont_coords[i], rinv),
      );
      if (!pt.equals(zero)) { pt.assertValidity(); }
      m_points.push(pt);
    }

    // Convert the reduced buckets out of Montgomery form
    const g_points: ExtPointType[] = []
    for (let i = 0; i < num_threads; i ++) {
      const idx = subtask_idx * num_threads + i
      const pt = fieldMath.createPoint(
        fieldMath.Fp.mul(g_points_x_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_y_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_t_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_z_mont_coords[idx], rinv),
      );
      if (!pt.equals(zero)) { pt.assertValidity(); }
      g_points.push(pt);
    }

    const expected = parallel_bucket_reduction_1(original_bucket_sums, num_threads)
    for (let i = 0; i < expected.g_points.length; i ++) {
      assert(g_points[i].equals(expected.g_points[i]), `mismatch at ${i}`)
    }
  }
}

const bpr_2 = async (
  shaderCode: string,
  subtask_idx: number,
  num_subtasks: number,
  num_x_workgroups: number,
  workgroup_size: number,
  num_columns: number,
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  bucket_sum_x_sb: GPUBuffer,
  bucket_sum_y_sb: GPUBuffer,
  bucket_sum_t_sb: GPUBuffer,
  bucket_sum_z_sb: GPUBuffer,
  g_points_x_sb: GPUBuffer,
  g_points_y_sb: GPUBuffer,
  g_points_t_sb: GPUBuffer,
  g_points_z_sb: GPUBuffer,
  debug = false,
) => {
  // Parameters as a uniform buffer
  const params_bytes = numbers_to_u8s_for_gpu([
    subtask_idx,
    num_columns,
  ]);
  const params_ub = create_and_write_ub(device, params_bytes);

  const bindGroupLayout = create_bind_group_layout(device, [
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "storage",
    "uniform",
  ]);

  const bindGroup = create_bind_group(device, bindGroupLayout, [
    bucket_sum_x_sb,
    bucket_sum_y_sb,
    bucket_sum_t_sb,
    bucket_sum_z_sb,
    g_points_x_sb,
    g_points_y_sb,
    g_points_t_sb,
    g_points_z_sb,
    params_ub,
  ]);

  const computePipeline = await create_compute_pipeline(
    device,
    [bindGroupLayout],
    shaderCode,
    "main_2",
  );

  const num_threads = num_x_workgroups * workgroup_size
  const num_y_workgroups = 1
  const num_z_workgroups = 1
  //console.log({ num_threads, num_x_workgroups, workgroup_size })

  execute_pipeline(
    commandEncoder,
    computePipeline,
    bindGroup,
    num_x_workgroups,
    num_y_workgroups,
    num_z_workgroups,
  );

  if (debug) {
    const data = await read_from_gpu(device, commandEncoder, [
      bucket_sum_x_sb,
      bucket_sum_y_sb,
      bucket_sum_t_sb,
      bucket_sum_z_sb,
      g_points_x_sb,
      g_points_y_sb,
      g_points_t_sb,
      g_points_z_sb,
    ])

    // The number of buckets per subtask
    const n = num_columns / 2

    const start = subtask_idx * n * num_words * 4
    const end = (subtask_idx * n + n) * num_words * 4

    const m_points_x_mont_coords = u8s_to_bigints(data[0].slice(start, end), num_words, word_size);
    const m_points_y_mont_coords = u8s_to_bigints(data[1].slice(start, end), num_words, word_size);
    const m_points_t_mont_coords = u8s_to_bigints(data[2].slice(start, end), num_words, word_size);
    const m_points_z_mont_coords = u8s_to_bigints(data[3].slice(start, end), num_words, word_size);

    const g_points_x_mont_coords = u8s_to_bigints(data[4], num_words, word_size);
    const g_points_y_mont_coords = u8s_to_bigints(data[5], num_words, word_size);
    const g_points_t_mont_coords = u8s_to_bigints(data[6], num_words, word_size);
    const g_points_z_mont_coords = u8s_to_bigints(data[7], num_words, word_size);

    const zero = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const m_points: ExtPointType[] = []
    for (let i = 0; i < n; i ++) {
      const pt = fieldMath.createPoint(
        fieldMath.Fp.mul(m_points_x_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_y_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_t_mont_coords[i], rinv),
        fieldMath.Fp.mul(m_points_z_mont_coords[i], rinv),
      );
      if (!pt.equals(zero)) { pt.assertValidity(); }
      m_points.push(pt);
    }

    // Convert the reduced buckets out of Montgomery form
    const g_points: ExtPointType[] = []
    for (let i = 0; i < num_threads; i ++) {
      const idx = subtask_idx * num_threads + i
      const pt = fieldMath.createPoint(
        fieldMath.Fp.mul(g_points_x_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_y_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_t_mont_coords[idx], rinv),
        fieldMath.Fp.mul(g_points_z_mont_coords[idx], rinv),
      );
      if (!pt.equals(zero)) { pt.assertValidity(); }
      g_points.push(pt);
    }

    const expected = parallel_bucket_reduction_2(g_points, m_points, n, num_threads)
    // TODO: figure out why the following fails at index 0
    for (let i = 0; i < expected.length; i ++) {
      assert(g_points[i].equals(expected[i]), `mismatch at ${i}`)
    }
  }
}
