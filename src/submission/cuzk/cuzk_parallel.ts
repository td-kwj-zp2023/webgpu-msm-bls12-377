import { ExtPointType } from "@noble/curves/abstract/edwards";
import { BigIntPoint } from "../../reference/types";
import { CSRSparseMatrix, ELLSparseMatrix, fieldMath } from '../matrices/matrices';
import { webWorkers } from "./workers/worker";

export async function init(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<CSRSparseMatrix[]> {  
    // λ-bit scalars
    const lambda = 256
  
    // s-bit window size 
    const s = 16
  
    // Thread count
    const threads = 16
  
    // Number of rows and columns (ie. row-space)
    const num_rows = threads
    const num_columns = Math.pow(2, s) - 1
  
    // Intantiate empty array of sparse matrices
    const csr_sparse_matrix_array: CSRSparseMatrix[] = []
    
    const ZERO_POINT = fieldMath.customEdwards.ExtendedPoint.ZERO;
    for (let i = 0; i < num_rows; i++) {
      // Instantiate empty ELL sparse matrix format
      const data = new Array(num_rows);
      for (let i = 0; i < num_rows; i++) {
          data[i] = new Array(num_columns).fill(ZERO_POINT);
      }
  
      const col_idx = new Array(num_rows);
      for (let i = 0; i < num_rows; i++) {
          col_idx[i] = new Array(num_columns).fill(0);
      }
  
      const row_length = Array(num_rows).fill(0);
  
      // Perform scalar decomposition
      const scalars_decomposed: bigint[][] = []
      for (let j =  Math.ceil(lambda / s); j > 0; j--) {
        const chunk: bigint[] = [];
        for (let i = 0; i < scalars.length; i++) {
          const mask = (BigInt(1) << BigInt(s)) - BigInt(1)  
          const limb = (scalars[i] >> BigInt(((j - 1) * s))) & mask // Right shift and extract lower 32-bits 
          chunk.push(limb)
        }
        scalars_decomposed.push(chunk);
      }
      
      // Divide EC points into t parts
      for (let thread_idx = 0; thread_idx < num_rows; thread_idx++) {
        for (let j = 0; j < num_columns; j++) {
            const point_i = thread_idx + j * threads
            data[thread_idx][j] = baseAffinePoints[point_i]
            col_idx[thread_idx][j] = scalars_decomposed[i][point_i]
            row_length[thread_idx] += 1
        }
      }
      
      // Transform ELL sparse matrix to CSR sparse matrix
      const ell_sparse_matrix = new ELLSparseMatrix(data, col_idx, row_length)
      const csr_sparse_matrix = await new CSRSparseMatrix([], [], []).ell_to_csr_sparse_matrix(ell_sparse_matrix)
  
      csr_sparse_matrix_array.push(csr_sparse_matrix)
    }
  
    return csr_sparse_matrix_array 
  }

export async function execute_cuzk_parallel(
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<ExtPointType> {    
    const inputSize = baseAffinePoints.length

    // Initialize instance 
    const csr_sparse_matrices = await init(baseAffinePoints, scalars)

    // Use `hardwareConcurrency` instead
    const maxWebWorkers = 8; 

    // Array of web worker promises
    const workerPromises = [];

    // Execute 2 rounds of 8 concurrent web workers each
    for (let i = 0; i < maxWebWorkers; i++) {
        workerPromises.push(webWorkers(csr_sparse_matrices[i]))
    }
    const results: ExtPointType[] = await Promise.all(workerPromises);
    
    for (let i = maxWebWorkers; i < maxWebWorkers * 2; i++) {
        workerPromises.push(webWorkers(csr_sparse_matrices[i]))
    }
    const results1: ExtPointType[] = await Promise.all(workerPromises);

    // Serialize results to points
    const G: ExtPointType[] = []
    for (let i = 0; i < results1.length; i++) {
        G.push(fieldMath.createPoint(results1[i].ex, results1[i].ey, results1[i].et, results1[i].ez))
    }

    // Perform Honer's rule
    let T = G[0];
    for (let i = 1; i < G.length; i++) {
        T = T.multiply(BigInt(Math.pow(2, 16)));
        T = T.add(G[i]);
    }
  
    return T
}
