import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix } from './matrices/matrices'; 

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

/*
 * @param: points All the input points of the MSM.
 * @param: new_point_indices The output of a prep function, such as prep_for_cluster_method
 * @param: cluster_start_indices The output of a prep function, such as prep_for_cluster_method
 */
export const pre_aggregate_cpu = (
    points: ExtPointType[],
    new_point_indices: number[],
    cluster_start_indices: number[],
) => {
    // Not the point at infinity!
    const ZERO_POINT = fieldMath.createPoint(
        BigInt(0),
        BigInt(0),
        BigInt(0),
        BigInt(0),
    )

    // Copy the points into a new array as we don't want to overwrite the
    // original
    //const new_points = points.map((x) => x)
    const new_points: ExtPointType[] = []

    for (let i = 0; i < cluster_start_indices.length; i ++) {
        // Example: [0, 1, 4, 7]
        // Case 0: 0, 1
        // Case 1: 1, 4
        // Case 2: 4, 7
        // Case 3: 7, end
        
        const start_idx = cluster_start_indices[i]
        let end_idx
        if (i === cluster_start_indices.length - 1) {
            // Case 3: we've reached the end of the array
            end_idx = cluster_start_indices.length
        } else {
            // Cases 1, 2: end_idx is the next index
            end_idx = cluster_start_indices[i + 1]
        }

        let acc = points[new_point_indices[start_idx]]
        new_points.push(acc)

        // This for loop won't execute for Case 1 because 0 + 1 is not smaller
        // than 1
        for (let j = start_idx + 1; j < end_idx; j ++) {
            acc = acc.add(points[new_point_indices[j]])
            new_points.push(acc)
            new_points[j - 1] = ZERO_POINT
        }
    }
    return new_points
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
        const chunk = scalar_chunks[i]
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

    let prev_chunk = scalar_chunks[new_point_indices[0]]
    for (let i = 1; i < new_point_indices.length; i ++) {
        if (prev_chunk !== scalar_chunks[new_point_indices[i]]) {
            cluster_start_indices.push(i)
        }
        prev_chunk = scalar_chunks[new_point_indices[i]]
    }
    return { new_point_indices, cluster_start_indices }
}

// Compute a "plan" which helps the parent algo pre-aggregate the points which
// share the same scalar chunk.
export const gen_add_to = (
    chunks: number[]
): { add_to: number[], new_chunks: number[] } => {
    const new_chunks = chunks.map((x) => x)
    const occ = new Map()
    const track = new Map()
    for (let i = 0; i < chunks.length; i ++) {
        const chunk = chunks[i]
        if (occ.get(chunk) != undefined) {
            occ.get(chunk).push(i)
        } else {
            occ.set(chunk, [i])
        }

        track.set(chunk, 0)
    }

    const add_to = Array.from(new Uint8Array(chunks.length))
    for (let i = 0; i < chunks.length; i ++) {
        const chunk = chunks[i]
        const t = track.get(chunk)
        if (t === occ.get(chunk).length - 1 || chunk === 0) {
            continue
        }

        add_to[i] = occ.get(chunk)[t + 1]
        track.set(chunk, t + 1)
        new_chunks[i] = 0
    }

    // Sanity check
    assert(add_to.length === chunks.length)
    assert(add_to.length === new_chunks.length)

    return { add_to, new_chunks }
}

export function merge_points(
    points: ExtPointType[],
    add_to: number[],
    zero_point: ExtPointType,
) {
    // merged_points will contain points that have been accumulated based on common scalar chunks.
    // e.g. if points == [P1, P2, P3, P4] and scalar_chunks = [1, 1, 2, 3],
    // merged_points will equal [0, P1 + P2, P3, P4]
    const merged_points = points.map((x) => fieldMath.createPoint(x.ex, x.ey, x.et, x.ez))

    // Next, add up the points whose scalar chunks match
    for (let i = 0; i < add_to.length; i ++) {
        if (add_to[i] != 0) {
            const cur = merged_points[i]
            merged_points[add_to[i]] = merged_points[add_to[i]].add(cur)
            merged_points[i] = zero_point
        }
    }

    return merged_points
}

const fieldMath = new FieldMath()
const ZERO_POINT = fieldMath.createPoint(
    BigInt(0),
    BigInt(1),
    BigInt(0),
    BigInt(1),
)

export function create_ell(
    points: ExtPointType[],
    scalar_chunks: number[],
    num_threads: number,
) {
    const num_cols = scalar_chunks.length / num_threads
    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []

    for (let i = 0; i < num_threads; i ++) {
        // Take each num_thread-th chunk only (each row)
        const chunks: number[] = []
        for (let j = 0; j < num_cols; j ++) {
            const idx = i * num_cols + j
            const c = scalar_chunks[idx]
            chunks.push(c)
        }

        // Pre-aggregate points per row
        const { add_to, new_chunks } = gen_add_to(chunks)
        const merged_points = merge_points(
            points,
            add_to,
            ZERO_POINT,
        )

        const pt_row: ExtPointType[] = []
        const idx_row: number[] = []
        for (let j = 0; j < num_cols; j ++) {
            const point_idx = num_cols * i + j
            const pt = merged_points[point_idx]
            if (new_chunks[point_idx] !== 0) {
                pt_row.push(pt)
                idx_row.push(new_chunks[point_idx])
            }
        }
        data.push(pt_row)
        col_idx.push(idx_row)
        row_length.push(pt_row.length)
    }
    const ell_sm = new ELLSparseMatrix(data, col_idx, row_length)
    return ell_sm

    /*
    // Precompute the indices for the points to merge
    const { add_to, new_chunks } = gen_add_to(scalar_chunks)
    const merged_points = merge_points(
        points,
        add_to,
        ZERO_POINT,
    )

    // Create an ELL sparse matrix using merged_points and new_chunks
    const num_cols = points.length / num_threads
    const data: ExtPointType[][] = []
    const col_idx: number[][] = []
    const row_length: number[] = []
    
    for (let i = 0; i < num_threads; i ++) {
        const pt_row: ExtPointType[] = []
        const idx_row: number[] = []
        for (let j = 0; j < num_cols; j ++) {
            const point_idx = num_cols * i + j
            const pt = merged_points[point_idx]
            if (new_chunks[point_idx] !== 0) {
                pt_row.push(pt)
                idx_row.push(new_chunks[point_idx])
            }
        }
        data.push(pt_row)
        col_idx.push(idx_row)
        row_length.push(pt_row.length)
    }
    const ell_sm = new ELLSparseMatrix(data, col_idx, row_length)
    return ell_sm
    */
}
