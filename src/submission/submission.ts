import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { execute_cuzk } from "./cuzk/cuzk"

// Typescript implementation of cuZK
export const compute_cuzk_typescript = async (
  baseAffinePoints: BigIntPoint[],
  scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
  const inputSize = 16

  const result = await execute_cuzk(inputSize, baseAffinePoints, scalars)

  const result_affine = result.toAffine()
  const x = result_affine.x
  const y = result_affine.y

  return { x, y }
};

// WGSL implementation of cuZK
export const compute_cuzk_wgsl = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{x: bigint, y: bigint}> => {
  throw new Error("Not implemented");
};