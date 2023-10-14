import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { DenseMatrix, ELLSparseMatrix, CSRSparseMatrix } from '../matrices/matrices'; 
import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { strict as assert } from 'assert';

export async function init(
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<any> {
  // Define number of inputs
  const n = 8
  
  // Define number of threads (ie. lambda / s threads)
  const t = 4
  
  // Define window size 
  const s = 2
  
  // Define number of rows and columns (ie. row-space)
  const num_rows = t
  const num_columns = n / t

  // Instantiate 'FieldMath' object
  const fieldMath = new FieldMath();

  // Instantiate empty ELL sparse matrix 
  const data = new Array(num_rows);
  for (let i = 0; i < num_rows; i++) {
      data[i] = new Array(num_columns).fill(fieldMath.customEdwards.ExtendedPoint.ZERO);
  }

  const col_idx = new Array(num_rows);
  for (let i = 0; i < num_rows; i++) {
      col_idx[i] = new Array(num_columns).fill(0);
  }
  const row_length = Array(num_rows).fill(0);

  // Divide EC points into t parts, where each thread handles n / t points
  for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
      const z = 0
      for (let j = 0; j < num_columns; j++) {
          const point_i = thread_idx + j * t
          data[thread_idx][j] = baseAffinePoints[point_i]
          col_idx[thread_idx][j] = scalars[point_i]
          row_length[thread_idx] += 1
      }
  }

  // Transform ELL sparse matrix to CSR sparse matrix
  const ell_sparse_matrix = new ELLSparseMatrix(data, col_idx, row_length)
  const csr_sparse_matrix = await new CSRSparseMatrix([], [], []).ell_to_csr_sparse_matrix(ell_sparse_matrix)

  return { csr_sparse_matrix, fieldMath }
}

export async function transpose_and_spmv(csr_sparse_matrix: CSRSparseMatrix, fieldMath: FieldMath): Promise<any> {
  // Transpose CSR sparse matrix
  const csr_sparse_matrix_transposed = await csr_sparse_matrix.transpose()

  // Derive partial bucket sums by performing SMVP
  const vector_smvp: bigint[] = Array(csr_sparse_matrix_transposed.row_ptr.length - 1).fill(BigInt(1));
  const buckets_svmp: ExtPointType[] = await csr_sparse_matrix_transposed.smvp(vector_smvp, fieldMath)

  // Aggregate SVMP buckets with running sum 
  let aggregate_svmp: ExtPointType = fieldMath.customEdwards.ExtendedPoint.ZERO
  for (const [i, bucket] of buckets_svmp.entries()) {
    if (i == 0) {
        continue
    }
    aggregate_svmp = aggregate_svmp.add(bucket.multiply(BigInt(i)))
  }

  // Convert results to affine coordinates
  const cuzk_result_1 = aggregate_svmp.toAffine();
  
  return cuzk_result_1
}

export async function smtvp(csr_sparse_matrix: CSRSparseMatrix, fieldMath: FieldMath): Promise<any> {
  // Derive partial bucket sums by performing SMTVP
  const vector_smtvp: bigint[] = Array(csr_sparse_matrix.row_ptr.length - 1).fill(BigInt(1));
  const buckets_svtmp: ExtPointType[] = await csr_sparse_matrix.smtvp(vector_smtvp, fieldMath)

  // Aggregate SVTMP buckets with running sum 
  let aggregate_svtmp: ExtPointType = fieldMath.customEdwards.ExtendedPoint.ZERO
  for (const [i, bucket] of buckets_svtmp.entries()) {
    if (i == 0) {
        continue
    }
    aggregate_svtmp = aggregate_svtmp.add(bucket.multiply(BigInt(i)))
  }

  // Convert results to affine coordinates
  const cuzk_result_2 = aggregate_svtmp.toAffine();

  return cuzk_result_2
}

export async function cuzk_compute_msm(
    baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
    scalars: bigint[] | Uint32Array[]
): Promise<any> {    
  // Initialize instance
  let parameters = await init(baseAffinePoints, scalars)

  // Perform Transpose and SPMV 
  let cuzk_result_1 = await transpose_and_spmv(parameters.csr_sparse_matrix, parameters.fieldMath)

  // Perform SPTMV
  let cuzk_result_2 = await smtvp(parameters.csr_sparse_matrix, parameters.fieldMath)

  // Compare cuZK results
  assert(cuzk_result_1.x == cuzk_result_2.x)
  assert(cuzk_result_1.y == cuzk_result_2.y)

  return { cuzk_result_1, cuzk_result_2 }
};