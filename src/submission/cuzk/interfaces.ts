export interface DenseMatrix {
    transpose(): Promise<DenseMatrix>
    matrix_vec_mult(vec: number[]): Promise<number[]>
}

export interface ELLSparseMatrix {
    dense_to_sparse_matrix(dense_matrix: DenseMatrix): Promise<ELLSparseMatrix>
    // sparse_to_dense_matrix(sparse_matrix: ELLSparseMatrix): Promise<DenseMatrix>
}

export interface CSRSparseMatrix {
    ell_to_csr_sparse_matrix(ell_sparse_matrix: ELLSparseMatrix): Promise<CSRSparseMatrix>
    dense_to_sparse_matrix(dense_matrix: DenseMatrix): Promise<CSRSparseMatrix>
    // sparse_to_dense_matrix(sparse_matrix: CSRSparseMatrix): Promise<DenseMatrix>
    smtvp(vec: number[]): Promise<number[]>
    // smtvp_parallel(vec: number[]): Promise<number[]>
    // transpose(): Promise<CSRSparseMatrix>
}



