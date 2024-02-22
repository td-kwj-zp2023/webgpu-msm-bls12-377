import { F, G1 } from "@celo/bls12377js";
import bigInt from "big-integer";

export enum Curve {
  BLS12_377,
  Edwards_BLS12,
}

export const base_field_modulus: any = {};
base_field_modulus[Curve.BLS12_377] = BigInt(
  "0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001",
);
base_field_modulus[Curve.Edwards_BLS12] = BigInt(
  "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
);

export const BASE_FIELD = BigInt(
  "0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001",
);

export const createGeneratorPoint = () => {
  const x = F.fromString(
    "81937999373150964239938255573465948239988671502647976594219695644855304257327692006745978603320413799295628339695",
  );
  const y = F.fromString(
    "241266749859715473739788878240585681733927191168601896383759122102112907357779751001206799952863815012735208165030",
  );
  return G1.fromElements(x, y);
};

export function createBaseF(val: bigint) {
  return F.fromString(val.toString());
}

/*
 * Convert X, Y, and Z coordiantes into a BLS12-377 point. The result will be
 * in affine form. The procedure convert a projective point to affine form is
 * to multiply each coordiante by the inverse of Z. Since Z * inv(Z) = 1, we
 * can just use G1.fromElements(X * inv(Z), Y * inv(Z)).
 */
export const createAffinePoint = (x: bigint, y: bigint, z: bigint) => {
  let x_b = createBaseF(x);
  let y_b = createBaseF(y);
  const z_b = createBaseF(z);

  if (z_b.toBig().eq(0)) {
    const pt = G1.fromElements(
      createBaseF(BigInt(0)),
      createBaseF(BigInt(1)),
    ).scalarMult(bigInt(0));
    return pt;
  }

  const z_inv = z_b.inverse();
  x_b = x_b.multiply(z_inv);
  y_b = y_b.multiply(z_inv);

  const p = G1.fromElements(x_b, y_b);
  return p;
};

export const ZERO = createAffinePoint(BigInt(0), BigInt(1), BigInt(0));

export const negate = (pt: G1) => {
  if (pt.equals(ZERO)) {
    return pt;
  }
  return pt.negate();
};

export const scalarMult = (pt: G1, scalar: bigint) => {
  return pt.scalarMult(bigInt(scalar.toString()));
};

export const get_bigint_x_y = (pt: G1) => {
  const x: bigint = Object(pt.x().toBig())["value"];
  const y: bigint = Object(pt.y().toBig())["value"];
  return { x, y };
};

export const get_bigint_x_y_z = (pt: G1) => {
  const x: bigint = Object(pt.x().toBig())["value"];
  const y: bigint = Object(pt.y().toBig())["value"];
  const z: bigint = Object(pt.z().toBig())["value"];
  return { x, y, z };
};
