import assert from 'assert'
import { CSRSparseMatrix } from '../matrices/matrices'

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

    const cluster_start_indices: number[] = [0]
    const cluster_end_indices: number[] = []
    const new_point_indices: number[] = []

    for (const chunk of clusters.keys()) {
        const cluster = clusters.get(chunk)
        // append single-item clusters
        if (cluster.length === 1) {
            new_point_indices.push(cluster[0])
        } else {
            for (const c of cluster) {
                new_point_indices.unshift(c)
            }
        }
    }

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

    return { new_point_indices, cluster_start_indices, cluster_end_indices }
}


export const pre_aggregate = (
    points: any[],
    scalar_chunks: number[],
    new_point_indices: number[],
    cluster_start_indices: number[],
    cluster_end_indices: number[],
) =>  {
    const new_points: any[] = []
    const new_scalar_chunks: number[] = []
    for (let i = 0; i < cluster_start_indices.length; i ++) {
        const start_idx = cluster_start_indices[i]
        const end_idx = cluster_end_indices[i]

        let acc = points[new_point_indices[start_idx]]
        for (let j = start_idx + 1; j < end_idx; j ++) {
            acc += points[new_point_indices[j]]
        }

        new_points.push(acc)
        new_scalar_chunks.push(scalar_chunks[new_point_indices[start_idx]])
    }

    return { new_points, new_scalar_chunks }
}

export const create_csr = (
    points: any[],
    scalar_chunks: number[],
    num_rows: number,
) => {
    let all_new_point_indices: number[] = []
    let all_cluster_start_indices: number[] = []
    let all_cluster_end_indices: number[] = []

    for (let row_idx = 0; row_idx < num_rows; row_idx ++) {
        const { new_point_indices, cluster_start_indices, cluster_end_indices } = 
                precompute_with_cluster_method(scalar_chunks, row_idx, num_rows)

        for (let i = 0; i < cluster_start_indices.length; i ++) {
            cluster_start_indices[i] += all_new_point_indices.length
            cluster_end_indices[i] += all_new_point_indices.length
        }

        all_new_point_indices = all_new_point_indices.concat(new_point_indices)
        all_cluster_start_indices = all_cluster_start_indices.concat(cluster_start_indices)
        all_cluster_end_indices = all_cluster_end_indices.concat(cluster_end_indices)
    }

    const { new_points, new_scalar_chunks } = pre_aggregate(
        points,
        scalar_chunks,
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
    )
    const row_ptr: number[] = all_cluster_start_indices
    row_ptr.push(all_new_point_indices.length)

    return new CSRSparseMatrix(new_points, new_scalar_chunks, row_ptr)
}
