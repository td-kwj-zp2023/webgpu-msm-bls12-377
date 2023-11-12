import { BigIntPoint } from "../../reference/types"
import * as wasm from 'csr-precompute'

export async function create_csr_wasm_precomputation_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
	wasm.precompute_with_cluster_method()
    return { x: BigInt(0), y: BigInt(0) }
}
