export enum Curve {
    BLS12_377,
    Edwards_BLS12,
}

export const base_field_modulus: any = {}
base_field_modulus[Curve.BLS12_377] = BigInt('0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001')
base_field_modulus[Curve.Edwards_BLS12] = BigInt('0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001')

// TODO: generate constants for curves and word sizes and store them in a JSON

//export const params = {
    //b: {
        //13: {
            //r: BigInt(1),
            //rinv: BigInt(1),
            //n0: BigInt(1),
            //edwards_d: BigInt(1),
        //}
    //}
//}
