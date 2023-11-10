import { create_csr_cpu } from './create_csr'

describe('Create an CSR sparse matrix from the MSM input points and scalars', () => {
    describe('pre-aggregation using the cluster method', () => {
        describe('small tests', () => {
            const num_points = 8
            const num_rows = 2
            it('small test #1', () => {

                const points = []
                for (let i = 0; i < num_points; i ++) {
                    points.push(`P${i}`)
                }

                const decomposed_scalars = [
                    [
                        4, 4, 4, 3,
                        3, 3, 3, 0,
                    ],
                    [
                        3, 4, 4, 3,
                        4, 1, 0, 2,
                    ],
                ]

                const expected_data = [
                    ['P2P1P0', 'P6P5P4', 'P3'],
                    ['P2P1', 'P3P0', 'P4', 'P5', 'P7'],
                ]

                const expected_col_idxs = [
                    [4, 3, 3],
                    [4, 3, 4, 1, 2],
                ]

                const expected_row_ptrs = [
                    [0, 2, 3],
                    [0, 2, 5],
                ]

                for (let i = 0; i < decomposed_scalars.length; i ++) {
                    const scalar_chunks = decomposed_scalars[i]
                    const csr_sm = create_csr_cpu(
                        points,
                        scalar_chunks,
                        num_rows,
                        (a: string, b: string) => a + b
                    )
                    expect(csr_sm.data.toString()).toEqual(expected_data[i].toString())
                    expect(csr_sm.col_idx.toString()).toEqual(expected_col_idxs[i].toString())
                    expect(csr_sm.row_ptr.toString()).toEqual(expected_row_ptrs[i].toString())
                }

            })
        })
    })
})

export {}
