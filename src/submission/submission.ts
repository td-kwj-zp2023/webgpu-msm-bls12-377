import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { execute_cuzk } from "./cuzk/cuzk_serial"
import { execute_cuzk_parallel } from "./cuzk/cuzk_parallel"
import { execute_cuzk_wgsl } from "./cuzk/cuzk_wgsl"

// Typescript implementation of cuZK
export const compute_cuzk_typescript = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
  console.log("Starting Serial cuZK!")

  const inputSize = 16

  const result = await execute_cuzk(inputSize, baseAffinePoints, scalars)

  const result_affine = result.toAffine()
  const x = result_affine.x
  const y = result_affine.y

  return { x, y }
};

// Typescript implementation of cuZK with web-workers
export const compute_cuzk_typescript_web_workers = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<any> => {
  console.log("Starting Parallel cuZK!")

  const inputSize = 16

  const result = await execute_cuzk_parallel(inputSize, baseAffinePoints, scalars)

  // const result_affine = result.toAffine()
  // const x = result_affine.x
  // const y = result_affine.y

  // return { x, y }
};

// WGSL implementation of cuZK
export const compute_cuzk_wgsl = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
  console.log("Starting WGSL cuZK!")
  
  const inputSize = 16

  const result = await execute_cuzk_wgsl(inputSize, baseAffinePoints, scalars)

  throw new Error("Not implemented");
};