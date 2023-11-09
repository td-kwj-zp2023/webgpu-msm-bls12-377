import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix } from './matrices/matrices'; 

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

export const compute_new_scalar_chunks = (
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
) => {
    const new_scalar_chunks: number[] = []

    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        if (scalar_chunks[new_point_indices[start_idx]] !== 0) {
            new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
        }
    }
    return new_scalar_chunks
}

/*
 * @param: points All the input points of the MSM.
 * @param: new_point_indices The output of a prep function, such as prep_for_cluster_method
 * @param: cluster_start_indices The output of a prep function, such as prep_for_cluster_method
 * @return { new_points, new_scalar_chunks }: an array of points and associated scalar chunks
 * This function constructs a row of an ELL sparse matrix by pre-aggregating
 * points with the same scalar chunk.
 */
export const pre_aggregate_cpu = (
    points: ExtPointType[],
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
) => {
    const new_points: ExtPointType[] = []
    const new_scalar_chunks: number[] = []

    for (let i = 0; i < cluster_start_indices.length; i ++) {
        // Example: [0, 1, 4, 7]
        // Case 0: 0, 1
        // Case 1: 1, 4
        // Case 2: 4, 7
        // Case 3: 7, end
        
        const start_idx = cluster_start_indices[i]
        //if (scalar_chunks[new_point_indices[start_idx]] === 0) {
            //continue
        //}

        let end_idx
        if (i === cluster_start_indices.length - 1) {
            // Case 3: we've reached the end of the array
            end_idx = new_point_indices.length
            debugger
        } else {
            // Cases 1, 2: end_idx is the next index
            end_idx = cluster_start_indices[i + 1]
        }

        const p = points[new_point_indices[start_idx]]
        let acc = fieldMath.createPoint(p.ex, p.ey, p.et, p.ez)
        for (let j = start_idx + 1; j < end_idx; j ++) {
            const p = points[new_point_indices[j]]
            const pt = fieldMath.createPoint(p.ex, p.ey, p.et, p.ez)
            acc = acc.add(pt)
        }
        new_points.push(acc)
        new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
    }
    return { new_points, new_scalar_chunks }
}

/*
 * This "prep" method is meant to prepare inputs to assist the pre-aggregation
 * step. It sorts the scalar chunks, then calculates the cluster start indices.
 * e.g. if the scalar chunks are [3, 1, 3],
 * the sorted scalar chunks are [1, 3, 3], and
 * cluster_start_indices will be [0, 1, 2]
 * new_point_indices will be [1, 0, 2] 
 */
export const prep_for_sort_method = (
    scalar_chunks: number[],
    thread_idx: number,
    num_threads: number,
) => {
    assert(num_threads > 0)
    const pt_and_chunks = []
    const c = scalar_chunks.length / num_threads
    for (let i = 0; i < c; i ++) {
        const pt_idx = thread_idx * c + i
        pt_and_chunks.push([pt_idx, scalar_chunks[i]])
    }

    pt_and_chunks.sort((a: number[], b: number[]) => {
        if (a[1] > b[1]) { return 1 }
        else if (a[1] < b[1]) { return -1 }
        return 0
    })

    const cluster_start_indices = [0]
    let prev_chunk = pt_and_chunks[0][1]
    for (let k = 1; k < pt_and_chunks.length; k ++) {
        if (prev_chunk !== pt_and_chunks[k][1]) {
            cluster_start_indices.push(k)
        }
        prev_chunk = pt_and_chunks[k][1]
    }

    const new_point_indices = pt_and_chunks.map((x) => x[0])
    return { new_point_indices, cluster_start_indices }
}

/*
 * This "prep" method is meant to prepare inputs to assist the pre-aggregation
 * step. Instead of sorting the entire array, it just tallies the indices of
 * unique scalar chunks, then calculates the cluster start indices.
 * e.g. if the scalar chunks are [3, 1, 3],
 * the clustered scalar chunks are { 1: [1], 3: [0, 1]}
 * The clustered scalar chunks are then flattened, with clusters with more than
 * 1 element placed at the start of the array: [2, 0, 1]
 * cluster_start_indices will be [0, 1, 2]
 * new_point_indices will be [2, 0, 1] 
 *
 * This method is more efficient than prep_for_sort_method because the order of
 * the clusters does not matter. What does matter is that they are clustered.
 */
export const prep_for_cluster_method = (
    scalar_chunks: number[],
    thread_idx: number,
    num_threads: number,
) => {
    assert(num_threads > 0)
    const new_point_indices: number[] = []
    const cluster_start_indices: number[] = [0]
    const clusters = new Map()

    const c = scalar_chunks.length / num_threads
    for (let i = 0; i < c; i ++) {
        const pt_idx = thread_idx * c + i
        const chunk = scalar_chunks[pt_idx]
        const g = clusters.get(chunk)
        if (g == undefined) {
            clusters.set(chunk, [pt_idx])
        } else {
            g.push(pt_idx)
            clusters.set(chunk, g)
        }
    }

    for (const k of clusters.keys()) {
        const cluster = clusters.get(k)
        if (cluster.length === 1) {
            new_point_indices.push(cluster[0])
        } else {
            for (const c of cluster) {
                new_point_indices.unshift(c)
            }
        }
    }

    let row_last_end_idx = 1

    // Build cluster_start_indices
    let prev_chunk = scalar_chunks[new_point_indices[0]]
    for (let i = 1; i < new_point_indices.length; i ++) {
        if (prev_chunk !== scalar_chunks[new_point_indices[i]]) {
            cluster_start_indices.push(i)
        }
        prev_chunk = scalar_chunks[new_point_indices[i]]
        row_last_end_idx ++
    }

    return { new_point_indices, cluster_start_indices, row_last_end_idx }
}

const fieldMath = new FieldMath()

export function create_ell(
    points: ExtPointType[],
    scalar_chunks: number[],
    num_threads: number,
) {
    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
        const { new_point_indices, cluster_start_indices } = prep_for_cluster_method(
            scalar_chunks,
            thread_idx,
            num_threads,
        )
        const { new_points, new_scalar_chunks } = pre_aggregate_cpu(
            points, 
            scalar_chunks,
            new_point_indices,
            cluster_start_indices,
        )

        const pt_row: ExtPointType[] = []
        const idx_row: number[] = []

        assert(new_points.length === new_scalar_chunks.length)
        for (let j = 0; j < new_points.length; j ++) {
            //assert(new_scalar_chunks[j] !== 0)
            pt_row.push(new_points[j])
            idx_row.push(new_scalar_chunks[j])
        }

        data.push(pt_row)
        col_idx.push(idx_row)
        row_length.push(pt_row.length)
    }

    return new ELLSparseMatrix(data, col_idx, row_length)
}
