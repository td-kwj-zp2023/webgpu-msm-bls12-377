import mustache from 'mustache'
import { BigIntPoint } from "../reference/types"
import { ExtPointType } from "@noble/curves/abstract/edwards";
import {
    get_device,
    create_and_write_sb,
    create_bind_group,
    create_bind_group_layout,
    create_compute_pipeline,
    create_sb,
    read_from_gpu,
} from './gpu'
import structs from './wgsl/struct/structs.template.wgsl'
import bigint_funcs from './wgsl/bigint/bigint.template.wgsl'
import field_funcs from './wgsl/field/field.template.wgsl'
import ec_funcs from './wgsl/curve/ec.template.wgsl'
import barrett_functions from './wgsl/barrett.template.wgsl'
import montgomery_product_funcs from './wgsl/montgomery/mont_pro_product.template.wgsl'

import assert from 'assert'

export const bucket_points_reduction = async (
    baseAffinePoints: BigIntPoint[],
    scalars: bigint[]
): Promise<{x: bigint, y: bigint}> => {
    const input_size = 1024
    assert(baseAffinePoints.length >= input_size)
    const points: ExtPointType[] = []
    for (const pt of baseAffinePoints.slice(0, input_size)) {
    }
}
