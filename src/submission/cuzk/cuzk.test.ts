// import { assert } from "console"
// import { BigIntPoint, U32ArrayPoint } from "../../reference/types";
// import { bigIntToU32Array, generateRandomFields } from '../../reference/webgpu/utils';
// import { init, transpose_and_spmv, smtvp } from './cuzk_serial'
// import { fieldMath } from '../matrices/matrices'; 

// export const generate_points = async(inputSize: number): Promise<{
//     bigIntPoints: BigIntPoint[],
//     U32Points: U32ArrayPoint[],
//   }> => {
//     // Creating random points is slow, so for now use a single fixed base.
//     const x = BigInt('2796670805570508460920584878396618987767121022598342527208237783066948667246');
//     const y = BigInt('8134280397689638111748378379571739274369602049665521098046934931245960532166');
//     const t = BigInt('3446088593515175914550487355059397868296219355049460558182099906777968652023');
//     const z = BigInt('1');
//     const point: BigIntPoint = {x, y, t, z};
//     const bigIntPoints: BigIntPoint[] = Array(inputSize).fill(point);
//     const U32Points: U32ArrayPoint[] = bigIntPoints.map((point) => {
//         return {
//         x: bigIntToU32Array(point.x),
//         y: bigIntToU32Array(point.y),
//         t: bigIntToU32Array(point.t),
//         z: bigIntToU32Array(point.z),
//         }});

//     return { bigIntPoints, U32Points }
// }

// export const generate_scalars = async(inputSize: number): Promise<{
//     bigIntScalars: bigint[],
//     U32Scalars: Uint32Array[]
//   }> => {
//     const bigIntScalars: bigint[] = generateRandomFields(inputSize);
//     const U32Scalars: Uint32Array[] = bigIntScalars.map((scalar) => bigIntToU32Array(scalar));
//     return { bigIntScalars, U32Scalars }
// }

// describe('cuzk serial test', () => {
//     describe('', () => {
//         it('spmv_and_sptvm_test', async () => {
//             let inputSize = 16
//             let points = await generate_points(inputSize)
//             let scalars = await generate_scalars(inputSize)
            
//             // Define sample scalars
//             scalars.bigIntScalars[0] = BigInt(1155)
//             scalars.bigIntScalars[1] = BigInt(9206)
//             scalars.bigIntScalars[2] = BigInt(2050)
//             scalars.bigIntScalars[3] = BigInt(8173)
//             scalars.bigIntScalars[4] = BigInt(38313)
//             scalars.bigIntScalars[5] = BigInt(28598)
//             scalars.bigIntScalars[6] = BigInt(54472)
//             scalars.bigIntScalars[7] = BigInt(61523)
//             scalars.bigIntScalars[8] = BigInt(3823)
//             scalars.bigIntScalars[9] = BigInt(29232)
//             scalars.bigIntScalars[10] = BigInt(2934)
//             scalars.bigIntScalars[11] = BigInt(10239)
//             scalars.bigIntScalars[12] = BigInt(8374)
//             scalars.bigIntScalars[13] = BigInt(10384)
//             scalars.bigIntScalars[14] = BigInt(53621)
//             scalars.bigIntScalars[15] = BigInt(8372)

//             // Initialize instance 
//             const csr_sparse_matrix_array = await init(inputSize, points.bigIntPoints, scalars.bigIntScalars)

//             // Perform Transpose and SPMV 
//             const cuzk_result_1 = await transpose_and_spmv(csr_sparse_matrix_array[15])
//             console.log(cuzk_result_1)
            
//             // Perform SMTVP
//             const cuzk_result_2 = await smtvp(csr_sparse_matrix_array[15])
//             console.log(cuzk_result_2)

//             // Assertion checks
//             assert(cuzk_result_1.x === cuzk_result_2.x)
//             assert(cuzk_result_1.y === cuzk_result_2.y)
//         })
//     })
// })

export {}