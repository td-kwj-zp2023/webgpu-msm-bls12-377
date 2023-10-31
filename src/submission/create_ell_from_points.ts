import { to_words_le } from './utils'
import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { bigIntPointToExtPointType } from './utils'
import { FieldMath } from "../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

// e.g. if the scalars converted to limbs = [
//    [limb_a, limb_b],
//    [limb_c, limb_d]
// ]
// return: [
//    [limb_a, limb_c],
//    [limb_b, limb_d]
// ]
const decompose_scalars = (
    scalars: bigint[],
    num_words: number,
    word_size: number,
): number[][] => {
    const as_limbs: number[][] = []
    for (const scalar of scalars) {
        const limbs = to_words_le(scalar, num_words, word_size)
        as_limbs.push(Array.from(limbs))
    }
    const result: number[][] = []
    for (let i = 0; i < num_words; i ++) {
        const t = as_limbs.map((limbs) => limbs[i])
        result.push(t)
    }
    return result
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
    const merged_points = points.map((x) => x)

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

export function create_ell_sparse_matrices_from_points(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[],
    num_threads: number,
): ELLSparseMatrix[] {
    // The number of threads is the number of rows of the matrix
    // As such the number of threads should divide the number of points
    assert(baseAffinePoints.length % num_threads === 0)
    assert(baseAffinePoints.length === scalars.length)

    const fieldMath = new FieldMath()
    const ZERO_POINT = fieldMath.createPoint(
        BigInt(0),
        BigInt(1),
        BigInt(0),
        BigInt(1),
    )
    const num_words = 20
    const word_size = 13

    // Decompose scalars
    const decomposed_scalars = decompose_scalars(scalars, num_words, word_size)

    const ell_sms: ELLSparseMatrix[] = []

    // For each set of decomposed scalars (e.g. the 0th chunks of each scalar,
    // the 1th chunks, etc, generate an ELL sparse matrix
    for (let i = 0; i < decomposed_scalars.length; i ++) {
        const scalar_chunks = decomposed_scalars[i]
        // Precompute the indices for the points to merge
        const { add_to, new_chunks } = gen_add_to(scalar_chunks)
        const merged_points = merge_points(
            baseAffinePoints.map((x) => bigIntPointToExtPointType(x, fieldMath)),
            add_to,
            ZERO_POINT,
        )

        // Create an ELL sparse matrix using merged_points and new_chunks
        const num_cols = baseAffinePoints.length / num_threads
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
        ell_sms.push(ell_sm)
    }
    return ell_sms
}
