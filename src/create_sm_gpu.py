#!/usr/bin/env python3

def prep_with_cluster_method(
    scalar_chunks,
    row_idx,
    num_rows
):
    num_cols = int(len(scalar_chunks) / num_rows)
    clusters = dict()

    # keep track of each cluster
    for i in range(0, num_cols):
        pt_idx = row_idx * num_cols + i
        s = scalar_chunks[pt_idx]

        # skip 0s
        if s == 0:
            continue

        if s in clusters:
            clusters[s].append(pt_idx)
        else:
            clusters[s] = [pt_idx]

    cluster_start_indices = [0]
    cluster_end_indices = []
    new_point_indices = []

    for s, point_indices in clusters.items():
        # append single-item clusters
        if len(point_indices) == 1:
            new_point_indices.append(point_indices[0])
        else:
        # prepend single-item clusters
            new_point_indices = clusters[s] + new_point_indices

    # populate cluster_start_indices and cluster_end_indices
    prev_chunk = scalar_chunks[new_point_indices[0]]
    for i in range(1, len(new_point_indices)):
        s = scalar_chunks[new_point_indices[i]]
        if prev_chunk != scalar_chunks[new_point_indices[i]]:
            cluster_end_indices.append(i)
            cluster_start_indices.append(i)
        prev_chunk = s

    # the final cluster_end_index
    cluster_end_indices.append(len(new_point_indices))

    return new_point_indices, cluster_start_indices, cluster_end_indices


def create_ell_gpu(points, scalar_chunks, num_rows):
    all_new_point_indices = []
    all_cluster_start_indices = []
    all_cluster_end_indices = []

    for row_idx in range(0, num_rows):
        new_point_indices, cluster_start_indices, cluster_end_indices = \
                prep_with_cluster_method(scalar_chunks, row_idx, num_rows)

        for i in range(0, len(cluster_start_indices)):
            cluster_start_indices[i] += len(all_new_point_indices)
            cluster_end_indices[i] += len(all_new_point_indices)

        all_new_point_indices += new_point_indices
        all_cluster_start_indices += cluster_start_indices
        all_cluster_end_indices += cluster_end_indices

    new_points, new_scalar_chunks = pre_aggregate(
        points,
        scalar_chunks,
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
    )

    print(points)
    print("start:", all_cluster_start_indices)
    print("end:  ", all_cluster_end_indices)
    print('all_new_point_indices', all_new_point_indices)
    print("new_points:", new_points)
    print("scalar_chunks:", scalar_chunks)
    print("new_scalar_chunks:", new_scalar_chunks)
    print()


def pre_aggregate(
    points,
    scalar_chunks,
    new_point_indices,
    cluster_start_indices,
    cluster_end_indices
):
    new_points = []
    new_scalar_chunks = []
    for i in range(0, len(cluster_start_indices)):
        start_idx = cluster_start_indices[i]
        end_idx = cluster_end_indices[i]

        acc = points[new_point_indices[start_idx]]
        for j in range(start_idx + 1, end_idx):
            acc += points[new_point_indices[j]]

        new_points.append(acc)
        new_scalar_chunks.append(scalar_chunks[new_point_indices[start_idx]])

    return new_points, new_scalar_chunks


def run():
    num_points = 8
    num_rows = 2

    points = []
    for i in range(0, num_points):
        points.append("P" + str(i))

    decomposed_scalars = [
        # [4, 4, 4, 3, 3, 3, 3, 0, 3, 4, 4, 3, 4, 1, 0, 2],
        [4, 4, 4, 3, 3, 3, 3, 0],
        [3, 4, 4, 3, 4, 1, 0, 2],
        [1, 2, 1, 2, 3, 4, 3, 4],
    ]

    for scalar_chunks in decomposed_scalars:
        create_ell_gpu(points, scalar_chunks, num_rows)


if __name__ == "__main__":
    run()
