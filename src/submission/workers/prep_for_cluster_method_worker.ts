import { expose } from 'threads/worker'

import { prep_for_cluster_method } from '../create_ell'

const handle_scalar_chunks = (
    scalar_chunks: number[],
    num_threads: number,
) => { 
    for (let thread_idx = 0; thread_idx < num_threads; thread_idx ++) {
        prep_for_cluster_method(scalar_chunks, thread_idx, num_threads)
    }
}

expose(handle_scalar_chunks)
