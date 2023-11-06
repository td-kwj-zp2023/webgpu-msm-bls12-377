import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { execute_cuzk } from "./cuzk/cuzk_serial"
import { execute_cuzk_parallel } from "./cuzk/cuzk_parallel"
import { transpose } from "./cuzk/transpose_wgsl"
import { smtvp } from "./cuzk/smtvp_wgsl"
import { smvp } from "./cuzk/smvp_wgsl"

// Typescript implementation of cuZK
export const cuzk_typescript_serial = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<any> => {
  console.log("Starting Serial cuZK!")

  const result = await execute_cuzk(baseAffinePoints, scalars)

  const result_affine = result.toAffine()
  const x = result_affine.x
  const y = result_affine.y

  return { x, y }
};

// Typescript implementation of cuZK with web-workers
export const cuzk_typescript_web_workers = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
  console.log("Starting Parallel cuZK!")

  const result = await execute_cuzk_parallel(baseAffinePoints, scalars)

  const result_affine = result.toAffine()
  const x = result_affine.x
  const y = result_affine.y

  return { x, y }
};

// WGSL implementation of Sparse-Matrix Transpose
export const transpose_wgsl = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<any> => {
  console.log("Starting WGSL Sparse-Matrix Transpose!")
  
  const result = await transpose(baseAffinePoints, scalars)

  throw new Error("Not implemented");
};

// WGSL implementation of Sparse-Matrix Vector Multiplication
export const smvp_wgsl = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<any> => {
  console.log("Starting WGSL SMVP!")
  
  const result = await smvp(baseAffinePoints, scalars)

  throw new Error("Not implemented");
};

// WGSL implementation of Sparse-Matrix Transpose
export const smtvp_wgsl = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<any> => {
  console.log("Starting WGSL SMTVP!")
  
  const result = await smtvp(baseAffinePoints, scalars)

  throw new Error("Not implemented");
};