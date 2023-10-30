import { to_words_le } from './utils'
import { BigIntPoint } from "../reference/types"
import { ELLSparseMatrix } from './matrices/matrices'; 
import { bigIntPointToExtPointType } from './utils'
import { FieldMath } from "../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import assert from 'assert'

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

    return { add_to, new_chunks }
}

export function merge_points(
    points: ExtPointType[],
    add_to: number[],
    zero_point: ExtPointType,
) {
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
    assert(baseAffinePoints.length === scalars.length)
    assert(baseAffinePoints.length % num_threads === 0)

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

        // merged_points contains the baseAffinePoints, but those whose scalar
        // chunks match have been accumulated. For example
        // if baseAffinePoints == [P1, P2, P3, P4] and scalar_chunks = [1, 1, 2, 3],
        // merged_points will equal [P1 + P2, 0, P3, P4]
        const merged_points: ExtPointType[] = []
        for (let j = 0; j < baseAffinePoints.length; j ++) {
            merged_points.push(bigIntPointToExtPointType(baseAffinePoints[j], fieldMath))
        }

        // Next, add up the points whose scalar chunks match
        for (let j = 0; j < add_to.length; j ++) {
            if (add_to[j] != 0) {
                const cur = merged_points[j]
                merged_points[add_to[j]] = merged_points[add_to[j]].add(cur)
                merged_points[j] = ZERO_POINT
            }
        }

        // Create an ELL sparse matrix using merged_points and new_chunks
        const data: ExtPointType[][] = []
        const col_idx: number[][] = []
        const row_length: number[] = []
        for (let i = 0; i < num_threads; i ++) {
            const col_space = baseAffinePoints.length / num_threads
            const pt_row: ExtPointType[] = []
            const idx_row: number[] = []
            let num_non_zero = 0
            for (let j = 0; j < col_space; j ++) {
                const point_idx = col_space * i + j
                const pt = merged_points[point_idx]
                if (new_chunks[point_idx] !== 0) {
                    pt_row.push(pt)
                    idx_row.push(new_chunks[point_idx])
                    num_non_zero ++
                }
            }
            data.push(pt_row)
            col_idx.push(idx_row)
            row_length.push(num_non_zero)
        }
        const ell_sm = new ELLSparseMatrix(data, col_idx, row_length)
        ell_sms.push(ell_sm)
    }
    return ell_sms
}
