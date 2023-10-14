import { assert } from "console"
import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
import { bigIntToU32Array, generateRandomFields } from '../../reference/webgpu/utils';
import { wasm_compute_msm } from '../../reference/reference'
import { cuzk_compute_msm } from './cuzk'

export const generate_points = async(inputSize: number): Promise<{
    bigIntPoints: BigIntPoint[],
    U32Points: U32ArrayPoint[],
  }> => {
    // Creating random points is slow, so for now use a single fixed base.
    const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246');
    const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166');
    const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023');
    const z = BigInt('1');
    const point: BigIntPoint = {x, y, t, z};
    const bigIntPoints: BigIntPoint[] = Array(inputSize).fill(point);
    const U32Points: U32ArrayPoint[] = bigIntPoints.map((point) => {
        return {
        x: bigIntToU32Array(point.x),
        y: bigIntToU32Array(point.y),
        t: bigIntToU32Array(point.t),
        z: bigIntToU32Array(point.z),
        }});

    return { bigIntPoints, U32Points }
}

export const generate_scalars = async(inputSize: number): Promise<{
    bigIntScalars: bigint[],
    U32Scalars: Uint32Array[]
  }> => {
    const bigIntScalars: bigint[] = generateRandomFields(inputSize);
    const U32Scalars: Uint32Array[] = bigIntScalars.map((scalar) => bigIntToU32Array(scalar));
    return { bigIntScalars, U32Scalars }
}

describe('cuzk test', () => {
    describe('', () => {
        it('spmv_and_sptvm_test', async () => {
            let inputSize = 8
            let points = await generate_points(inputSize)
            let scalars = await generate_scalars(inputSize)

            // Define sample scalars
            scalars.bigIntScalars[0] = BigInt(4)
            scalars.bigIntScalars[1] = BigInt(5)
            scalars.bigIntScalars[2] = BigInt(6)
            scalars.bigIntScalars[3] = BigInt(7)
            scalars.bigIntScalars[4] = BigInt(8)
            scalars.bigIntScalars[5] = BigInt(9)
            scalars.bigIntScalars[6] = BigInt(10)
            scalars.bigIntScalars[7] = BigInt(11)

            // Compute cuzk typescript MSM
            const cuzk_result = await cuzk_compute_msm(points.bigIntPoints, scalars.bigIntScalars)

            // Assertion checks
            assert(cuzk_result[0] === cuzk_result.x)
            assert(cuzk_result[1] === cuzk_result.y)
        })
    })
})