import { FieldMath } from "../reference/utils/FieldMath";
import { ELLSparseMatrix } from './matrices/matrices'; 

import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

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
