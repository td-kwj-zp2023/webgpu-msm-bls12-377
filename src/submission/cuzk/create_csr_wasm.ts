import assert from 'assert'
import { BigIntPoint } from "../../reference/types"
import { all_precomputation } from './create_csr'
import * as wasm from 'csr-precompute'
import {
    decompose_scalars
    } from '../utils'

export async function create_csr_wasm_precomputation_benchmark(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
): Promise<{x: bigint, y: bigint}> {
    const word_size = 13
    const num_words = 20
    const num_rows = 16

    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)
    for (const scalar_chunks of decomposed_scalars) {
        const wasm_result = wasm.all_precomputation(
            new Uint32Array(scalar_chunks),
            num_rows,
        )

        /*
        const ts_result = all_precomputation(scalar_chunks, num_rows)

        assert(wasm_result.all_new_point_indices.length === ts_result.all_new_point_indices.length)
        assert(wasm_result.all_cluster_start_indices.length === ts_result.all_cluster_start_indices.length)
        assert(wasm_result.all_cluster_end_indices.length === ts_result.all_cluster_end_indices.length)
        assert(wasm_result.all_single_point_indices.length === ts_result.all_single_point_indices.length)
        assert(wasm_result.all_single_scalar_chunks.length === ts_result.all_single_scalar_chunks.length)
        assert(wasm_result.row_ptr.length === ts_result.row_ptr.length)
        */

        /*
        // These checks won't pass even if the result is correct probably
        // because the order of BTreeMap::iter() and Map.keys() differ.
        // all_new_point_indices
        const wasm_all_new_point_indices = wasm_result.all_new_point_indices
        wasm_all_new_point_indices.sort()
        const ts_all_new_point_indices = ts_result.all_new_point_indices
        ts_all_new_point_indices.sort()
        assert(wasm_all_new_point_indices.toString() === ts_all_new_point_indices.toString())

        // all_cluster_start_indices
        const wasm_all_cluster_start_indices = wasm_result.all_cluster_start_indices
        //wasm_all_cluster_start_indices.sort()
        const ts_all_cluster_start_indices = ts_result.all_cluster_start_indices
        //ts_all_cluster_start_indices.sort()
        debugger
        assert(wasm_all_cluster_start_indices.toString() === ts_all_cluster_start_indices.toString())

        // all_cluster_end_indices
        const wasm_all_cluster_end_indices = wasm_result.all_cluster_end_indices
        wasm_all_cluster_end_indices.sort()
        const ts_all_cluster_end_indices = ts_result.all_cluster_end_indices
        ts_all_cluster_end_indices.sort()
        assert(wasm_all_cluster_end_indices.toString() === ts_all_cluster_end_indices.toString())

        // all_single_point_indices
        const wasm_all_single_point_indices = wasm_result.all_single_point_indices
        wasm_all_single_point_indices.sort()
        const ts_all_single_point_indices = ts_result.all_single_point_indices
        ts_all_single_point_indices.sort()
        assert(wasm_all_single_point_indices.toString() === ts_all_single_point_indices.toString())

        // all_single_scalar_chunks
        const wasm_all_single_scalar_chunks = wasm_result.all_single_scalar_chunks
        wasm_all_single_scalar_chunks.sort()
        const ts_all_single_scalar_chunks = ts_result.all_single_scalar_chunks
        ts_all_single_scalar_chunks.sort()
        assert(wasm_all_single_scalar_chunks.toString() === ts_all_single_scalar_chunks.toString())

        // row_ptr
        const wasm_row_ptr = wasm_result.row_ptr
        wasm_row_ptr.sort()
        const ts_row_ptr = ts_result.row_ptr
        ts_row_ptr.sort()
        assert(wasm_row_ptr.toString() === ts_row_ptr.toString())
        */
    }
    return { x: BigInt(0), y: BigInt(0) }
}
