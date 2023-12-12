import assert from 'assert'
import { CSRSparseMatrix } from '../matrices/matrices'; 
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const precompute_with_cluster_method = (
    scalar_chunks: number[],
    row_idx: number,
    num_rows: number,
) => {
    assert(scalar_chunks.length % num_rows === 0)
    const num_cols = scalar_chunks.length / num_rows
    const clusters = new Map()

    // keep track of each cluster
    for (let i = 0; i < num_cols; i ++ ) {
        const pt_idx = row_idx * num_cols + i
        const chunk = scalar_chunks[pt_idx]

        // skip 0s
        if (chunk === 0) {
            continue
        }

        const g = clusters.get(chunk)
        if (g == undefined) {
            clusters.set(chunk, [pt_idx])
        } else {
            g.push(pt_idx)
            clusters.set(chunk, g)
        }
    }

    let cluster_start_indices: number[] = [0]
    let cluster_end_indices: number[] = []
    let new_point_indices: number[] = []
    const s: number[] = []

    for (const chunk of clusters.keys()) {
        const cluster = clusters.get(chunk)
        if (cluster.length === 1) {
            s.push(cluster[0])
        } else {
            new_point_indices = new_point_indices.concat(cluster)
        }
    }
    // append single-item clusters
    new_point_indices = new_point_indices.concat(s)

    // populate cluster_start_indices and cluster_end_indices
    let prev_chunk = scalar_chunks[new_point_indices[0]]
    for (let i = 1; i < new_point_indices.length; i ++) {
        const s = scalar_chunks[new_point_indices[i]]
        if (prev_chunk != scalar_chunks[new_point_indices[i]]) {
            cluster_end_indices.push(i)
            cluster_start_indices.push(i)
        }
        prev_chunk = s
    }

    // the final cluster_end_index
    cluster_end_indices.push(new_point_indices.length)

    let i = 0
    while (i < cluster_start_indices.length) {
        if (cluster_start_indices[i] + 1 === cluster_end_indices[i]) {
            break
        }
        i ++
    }

    const num_non_zero = cluster_start_indices.length
    const singles_start_idx = i < cluster_start_indices.length ? [cluster_start_indices[i]] : []

    cluster_start_indices = cluster_start_indices.slice(0, i)
    cluster_end_indices = cluster_end_indices.slice(0, i)

    return { new_point_indices, cluster_start_indices, cluster_end_indices, singles_start_idx, num_non_zero }
}


export function pre_aggregate<P> (
    points: P[],
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
    cluster_end_indices: number[],
    add_func: (a: P, b: P) => P = (a: any, b: any) => a + b,
)  {
    const new_points: any[] = []
    const new_scalar_chunks: number[] = []
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        const end_idx = cluster_end_indices[i]

        let acc = points[new_point_indices[start_idx]]
        for (let j = start_idx + 1; j < end_idx; j ++) {
            acc = add_func(acc, points[new_point_indices[j]])
        }

        new_points.push(acc)
        new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
    }

    return { new_points, new_scalar_chunks }
}

export const all_precomputation = (
    scalar_chunks: number[],
    num_rows: number,
) => {
    let all_new_point_indices: number[] = []
    let all_cluster_start_indices: number[] = []
    let all_cluster_end_indices: number[] = []
    let all_single_point_indices: any[] = []
    let all_single_scalar_chunks: number[] = []

    const row_ptr: number[] = [0]

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { 
            new_point_indices,
            cluster_start_indices,
            cluster_end_indices,
            singles_start_idx,
            num_non_zero,
        } = precompute_with_cluster_method(scalar_chunks, row_idx, num_rows)

        row_ptr.push(row_ptr[row_ptr.length - 1] + num_non_zero)

        const single_point_indices: number[] = []
        const single_scalar_chunks: number[] = []

        if (singles_start_idx.length !== 0) {
            for (let i = singles_start_idx[0]; i < new_point_indices.length; i ++) {
                single_point_indices.push(new_point_indices[i])
                single_scalar_chunks.push(scalar_chunks[new_point_indices[i]])
            }
        }

        all_single_point_indices = all_single_point_indices.concat(single_point_indices)
        all_single_scalar_chunks = all_single_scalar_chunks.concat(single_scalar_chunks)

        for (let i = 0; i < cluster_start_indices.length; i ++) {
            cluster_start_indices[i] += all_new_point_indices.length
            cluster_end_indices[i] += all_new_point_indices.length
        }

        all_new_point_indices = all_new_point_indices.concat(new_point_indices)
        all_cluster_start_indices = all_cluster_start_indices.concat(cluster_start_indices)
        all_cluster_end_indices = all_cluster_end_indices.concat(cluster_end_indices)
    }

    return {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
        all_single_point_indices,
        all_single_scalar_chunks,
        row_ptr,
    }
}

export const create_csr_cpu = (
    points: any[],
    scalar_chunks: number[],
    num_rows: number,
    add_func: (a: any, b: any) => any = (a: any, b: any) => a.add(b),
) => {
    const {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
        all_single_point_indices,
        all_single_scalar_chunks,
        row_ptr,
    } = all_precomputation(scalar_chunks, num_rows)

    const { new_points, new_scalar_chunks } = pre_aggregate(
        points,
        scalar_chunks,
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
        add_func,
    )

    return new CSRSparseMatrix(
        new_points.concat(all_single_point_indices.map((x) => points[x])),
        new_scalar_chunks.concat(all_single_scalar_chunks),
        row_ptr,
    )
}
