import { FieldMath } from "../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";

export const add_points_any_a = (
    a: ExtPointType,
    b: ExtPointType,
    fieldMath: FieldMath,
): ExtPointType => {
    /*
    const A = modP(X1 * X2); // A = X1*X2
    const B = modP(Y1 * Y2); // B = Y1*Y2
    const C = modP(T1 * d * T2); // C = T1*d*T2
    const D = modP(Z1 * Z2); // D = Z1*Z2
    const E = modP((X1 + Y1) * (X2 + Y2) - A - B); // E = (X1+Y1)*(X2+Y2)-A-B
    const F = D - C; // F = D-C
    const G = D + C; // G = D+C
    const H = modP(B - a * A); // H = B-a*A
    const X3 = modP(E * F); // X3 = E*F
    const Y3 = modP(G * H); // Y3 = G*H
    const T3 = modP(E * H); // T3 = E*H
    const Z3 = modP(F * G); // Z3 = F*G
    return new Point(X3, Y3, Z3, T3);
    */
    const ed_a = BigInt('8444461749428370424248824938781546531375899335154063827935233455917409239040')
    const ed_d = BigInt(3021)
    const X1 = a.ex
    const Y1 = a.ey
    const T1 = a.et
    const Z1 = a.ez
    const X2 = b.ex
    const Y2 = b.ey
    const T2 = b.et
    const Z2 = b.ez

    const A = fieldMath.Fp.mul(X1, X2)
    const B = fieldMath.Fp.mul(Y1, Y2)
    const C = fieldMath.Fp.mul(fieldMath.Fp.mul(T1, ed_d), T2)
    const D = fieldMath.Fp.mul(Z1, Z2)
    let E = fieldMath.Fp.mul(
        fieldMath.Fp.add(X1, Y1),
        fieldMath.Fp.add(X2, Y2),
    )
    E = fieldMath.subtract(E, A)
    E = fieldMath.subtract(E, B)
    const F = fieldMath.Fp.sub(D, C)
    const G = fieldMath.Fp.add(D, C)
    const aA = fieldMath.Fp.mul(ed_a, A)
    const H = fieldMath.Fp.sub(B, aA)
    const X3 = fieldMath.Fp.mul(E, F); // X3 = E*F
    const Y3 = fieldMath.Fp.mul(G, H); // Y3 = G*H
    const T3 = fieldMath.Fp.mul(E, H); // T3 = E*H
    const Z3 = fieldMath.Fp.mul(F, G); // Z3 = F*G
    return fieldMath.createPoint(X3, Y3, T3, Z3)
}

export const add_points_a_minus_one = (
    a: ExtPointType,
    b: ExtPointType,
    fieldMath: FieldMath,
): ExtPointType => {
    const X1 = a.ex
    const Y1 = a.ey
    const T1 = a.et
    const Z1 = a.ez
    const X2 = b.ex
    const Y2 = b.ey
    const T2 = b.et
    const Z2 = b.ez

    const A = fieldMath.Fp.mul(
        fieldMath.Fp.sub(Y1, X1),
        fieldMath.Fp.add(Y2, X2)
    )
    const B = fieldMath.Fp.mul(
        fieldMath.Fp.add(Y1, X1),
        fieldMath.Fp.sub(Y2, X2)
    )
    const F = fieldMath.Fp.sub(B, A)
    const C = fieldMath.Fp.mul(fieldMath.Fp.mul(Z1, BigInt(2)), T2)
    const D = fieldMath.Fp.mul(fieldMath.Fp.mul(T1, BigInt(2)), Z2)
    const E = fieldMath.Fp.add(D, C)
    const G = fieldMath.Fp.add(B, A)
    const H = fieldMath.Fp.sub(D, C)
    const X3 = fieldMath.Fp.mul(E, F)
    const Y3 = fieldMath.Fp.mul(G, H)
    const T3 = fieldMath.Fp.mul(E, H)
    const Z3 = fieldMath.Fp.mul(F, G)
    return fieldMath.createPoint(X3, Y3, T3, Z3);
}
/*
add(other) {
    isPoint(other);
    const { a, d } = CURVE;
    const { ex: X1, ey: Y1, ez: Z1, et: T1 } = this;
    const { ex: X2, ey: Y2, ez: Z2, et: T2 } = other;
    // Faster algo for adding 2 Extended Points when curve's a=-1.
    // http://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
    // Cost: 8M + 8add + 2*2.
    // Note: It does not check whether the `other` point is valid.
    if (a === BigInt(-1)) {
        const A = modP((Y1 - X1) * (Y2 + X2));
        const B = modP((Y1 + X1) * (Y2 - X2));
        const F = modP(B - A);
        if (F === _0n)
            return this.double(); // Same point. Tests say it doesn't affect timing
        const C = modP(Z1 * _2n * T2);
        const D = modP(T1 * _2n * Z2);
        const E = D + C;
        const G = B + A;
        const H = D - C;
        const X3 = modP(E * F);
        const Y3 = modP(G * H);
        const T3 = modP(E * H);
        const Z3 = modP(F * G);
        return new Point(X3, Y3, Z3, T3);
    }
}
 */
