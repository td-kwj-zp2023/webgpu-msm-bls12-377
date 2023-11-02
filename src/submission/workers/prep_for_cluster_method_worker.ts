import { expose } from 'threads/worker'

import { prep_for_cluster_method } from '../create_ell'

expose(prep_for_cluster_method)
