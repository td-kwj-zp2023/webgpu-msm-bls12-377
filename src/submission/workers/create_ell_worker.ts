import { expose } from 'threads/worker'

import { create_ell } from '../create_ell'

expose(create_ell)
