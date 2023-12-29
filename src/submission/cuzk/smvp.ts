import { FieldMath } from "../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const cpu_smvp = (
    csc_col_ptr_sb: number[],
    points: ExtPointType[],
    fieldMath: FieldMath,
) => {
    const currentSum = fieldMath.customEdwards.ExtendedPoint.ZERO;

    const result: ExtPointType[] = [];
    for (let i = 0; i < csc_col_ptr_sb.length - 1; i++) {
        result.push(currentSum)
    }

    for (let i = 0; i < csc_col_ptr_sb.length - 1; i++) {
        const row_begin = csc_col_ptr_sb[i];
        const row_end = csc_col_ptr_sb[i + 1];
        let sum = currentSum;
        for (let j = row_begin; j < row_end; j++) {
            sum = sum.add(points[j])
        }
        result[i] = sum
    }

    return result
}
