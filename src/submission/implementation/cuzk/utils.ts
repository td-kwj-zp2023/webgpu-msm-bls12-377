/// This module provides various utility functions for cryptographic operations.

import assert from "assert";
import crypto from "crypto";
import * as bigintCryptoUtils from "bigint-crypto-utils";
import { BigIntPoint } from "../../../reference/types";
import { FieldMath } from "../../../reference/utils/FieldMath";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { EDWARDS_D } from "../../../reference/params/AleoConstants";

/*
 * Given a Buffer with values [a, b, c, d, ...] return a Buffer twice the size,
 * with pairs of 0s interspersed betwen each pair of values.
 * e.g. [a, b, 0, 0, c, d, 0, 0, ...]
 */
export const format_buffer_for_gpu = (buf: Buffer) => {
  const bytes = new Uint8Array(buf.length * 2)
  
  let k = 0
  for (let i = 0; i < buf.length; i += 2) {
    // Take each consecutive pair of values and space them out by two 0s
    const a = buf[i]
    const b = buf[i + 1]
    bytes[k] = a
    bytes[k + 1] = b
    k += 4
  }
  return bytes
}

export const format_points_buffer_for_gpu = (
  bufferPoints: Buffer,
  bytes_per_coord = 32,
) => {
  const input_size = bufferPoints.length / (bytes_per_coord * 2)
  const x_coords_bytes = new Uint8Array(bufferPoints.length)
  const y_coords_bytes = new Uint8Array(bufferPoints.length)

  // 32 bytes for extended twisted edwards, 48 for the BLS12-377 base field
  let k = 0
  for (let i = 0; i < input_size; i ++) {
    for (let j = 0; j < bytes_per_coord; j += 2) {
      x_coords_bytes[k]     = bufferPoints[i * bytes_per_coord * 2 + j]
      x_coords_bytes[k + 1] = bufferPoints[i * bytes_per_coord * 2 + j + 1]
      y_coords_bytes[k]     = bufferPoints[i * bytes_per_coord * 2 + bytes_per_coord + j]
      y_coords_bytes[k + 1] = bufferPoints[i * bytes_per_coord * 2 + bytes_per_coord + j + 1]
      k += 4
    }
  }
  return { x_coords_bytes, y_coords_bytes }
}

/*
 * Converts the BigInts in vals to byte arrays in the form of
 * [b0, b1, 0, 0, b2, b3, 0, 0, ...]
 * This is slower than bigints_to_u8_for_gpu, so don't use it
 */
export const bigints_to_16_bit_words_for_gpu = (vals: bigint[]): Uint8Array => {
  const result = new Uint8Array(64 * vals.length);
  for (let i = 0; i < vals.length; i++) {
    // This code snippet is adapted from
    // https://github.com/no2chem/bigint-buffer/blob/master/src/index.ts#L60
    const hex = vals[i].toString(16);
    const buf = Buffer.from(hex.padStart(128, "0").slice(0, 128), "hex");
    buf.reverse();

    for (let j = 0; j < buf.length; j += 2) {
      result[i * 64 + j * 2] = buf[j];
      result[i * 64 + j * 2 + 1] = buf[j + 1];
    }
  }
  return result;
};

export const bigIntPointToExtPointType = (
  bip: BigIntPoint,
  fieldMath: FieldMath,
): ExtPointType => {
  return fieldMath.createPoint(bip.x, bip.y, bip.t, bip.z);
};

export const extPointTypeToBigIntPoint = (ept: ExtPointType): BigIntPoint => {
  return { x: ept.ex, y: ept.ey, t: ept.et, z: ept.ez };
};

/**
 * e.g. if the scalars converted to limbs = [
 *          [limb_a, limb_b],
 *          [limb_c, limb_d]
 *      ]
 *      return: [
 *          [limb_a, limb_c],
 *          [limb_b, limb_d]
 *      ]
 */
export const decompose_scalars = (
  scalars: bigint[],
  num_words: number,
  word_size: number,
): number[][] => {
  const as_limbs: number[][] = [];
  for (const scalar of scalars) {
    const limbs = to_words_le(scalar, num_words, word_size);
    as_limbs.push(Array.from(limbs));
  }
  const result: number[][] = [];
  for (let i = 0; i < num_words; i++) {
    const t = as_limbs.map((limbs) => limbs[i]);
    result.push(t);
  }
  return result;
};

export const decompose_scalars_signed = (
  scalars: bigint[],
  num_words: number,
  word_size: number,
): number[][] => {
  const l = 2 ** word_size;
  const shift = 2 ** (word_size - 1);

  const as_limbs: number[][] = [];

  for (const scalar of scalars) {
    const limbs = to_words_le(scalar, num_words, word_size);
    const signed_slices: number[] = Array(limbs.length).fill(0);

    let carry = 0;
    for (let i = 0; i < limbs.length; i++) {
      signed_slices[i] = limbs[i] + carry;
      if (signed_slices[i] >= l / 2) {
        signed_slices[i] = (l - signed_slices[i]) * -1;
        if (signed_slices[i] === -0) {
          signed_slices[i] = 0;
        }

        carry = 1;
      } else {
        carry = 0;
      }
    }

    if (carry === 1) {
      console.error(scalar);
      throw new Error("final carry is 1");
    }

    as_limbs.push(Array.from(signed_slices).map((x) => x + shift));
  }

  const result: number[][] = [];
  for (let i = 0; i < num_words; i++) {
    const t = as_limbs.map((limbs) => limbs[i]);
    result.push(t);
  }
  return result;
};

export const points_to_u8s_for_gpu = (
  points: BigIntPoint[],
  num_words: number,
  word_size: number,
): Uint8Array => {
  const size = points.length * num_words * 4 * 4;
  const result = new Uint8Array(size);

  for (let i = 0; i < points.length; i++) {
    const x_bytes = bigint_to_u8_for_gpu(points[i].x, num_words, word_size);
    const y_bytes = bigint_to_u8_for_gpu(points[i].y, num_words, word_size);
    const t_bytes = bigint_to_u8_for_gpu(points[i].t, num_words, word_size);
    const z_bytes = bigint_to_u8_for_gpu(points[i].z, num_words, word_size);

    for (let j = 0; j < x_bytes.length; j++) {
      const i4l = i * 4 * x_bytes.length;
      result[i4l + j] = x_bytes[j];
      result[i4l + j + x_bytes.length] = y_bytes[j];
      result[i4l + j + x_bytes.length * 2] = t_bytes[j];
      result[i4l + j + x_bytes.length * 3] = z_bytes[j];
    }
  }

  return result;
};

export const u8s_to_points = (
  bytes: Uint8Array,
  num_words: number,
  word_size: number,
): BigIntPoint[] => {
  // Since each limb is a u32, there are 4 u8s per limb
  const num_u8s_per_coord = num_words * 4;
  const num_u8s_per_point = num_u8s_per_coord * 4;

  assert(bytes.length % num_u8s_per_point === 0);
  const result: BigIntPoint[] = [];
  for (let i = 0; i < bytes.length / num_u8s_per_point; i++) {
    const p = i * num_u8s_per_point;
    const x_i = p;
    const y_i = p + num_u8s_per_coord;
    const t_i = p + num_u8s_per_coord * 2;
    const z_i = p + num_u8s_per_coord * 3;

    const x_u8s = bytes.slice(x_i, x_i + num_u8s_per_coord);
    const y_u8s = bytes.slice(y_i, y_i + num_u8s_per_coord);
    const t_u8s = bytes.slice(t_i, t_i + num_u8s_per_coord);
    const z_u8s = bytes.slice(z_i, z_i + num_u8s_per_coord);

    const x = u8s_to_bigint(x_u8s, num_words, word_size);
    const y = u8s_to_bigint(y_u8s, num_words, word_size);
    const t = u8s_to_bigint(t_u8s, num_words, word_size);
    const z = u8s_to_bigint(z_u8s, num_words, word_size);

    result.push({ x, y, t, z });
  }

  return result;
};

export const u8s_to_bigints = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint[] => {
  const num_u8s_per_scalar = num_words * 4;
  const result = [];
  for (let i = 0; i < u8s.length / num_u8s_per_scalar; i++) {
    const p = i * num_u8s_per_scalar;
    const s = u8s.slice(p, p + num_u8s_per_scalar);
    result.push(u8s_to_bigint(s, num_words, word_size));
  }
  return result;
};

export const u8s_to_bigint = (
  u8s: Uint8Array,
  num_words: number,
  word_size: number,
): bigint => {
  const a = new Uint16Array(u8s.buffer);
  const limbs: number[] = [];
  for (let i = 0; i < a.length; i += 2) {
    limbs.push(a[i]);
  }

  return from_words_le(new Uint16Array(limbs), num_words, word_size);
};

export const numbers_to_u8s_for_gpu = (vals: number[]): Uint8Array => {
  // Expect each val to be max 32 bits
  const max = 2 ** 32;
  for (const val of vals) {
    assert(val < max);
  }
  const b = new Uint32Array(vals);
  return new Uint8Array(b.buffer);
};

export const u8s_to_numbers = (u8s: Uint8Array): number[] => {
  const result: number[] = [];
  assert(u8s.length % 4 === 0);
  for (let i = 0; i < u8s.length / 4; i++) {
    const n0 = u8s[i * 4];
    const n1 = u8s[i * 4 + 1];
    result.push(n1 * 256 + n0);
  }
  return result;
};

export const u8s_to_numbers_32 = (u8s: Uint8Array): number[] => {
  const result: number[] = [];
  assert(u8s.length % 4 === 0);
  for (let i = 0; i < u8s.length / 4; i++) {
    const n0 = u8s[i * 4];
    const n1 = u8s[i * 4 + 1];
    const n2 = u8s[i * 4 + 2];
    const n3 = u8s[i * 4 + 3];
    result.push(n3 * 16777216 + n2 * 65536 + n1 * 256 + n0);
  }
  return result;
};

// Assumes 32-byte BigInts
export const bigints_to_u8_for_gpu = (
  vals: bigint[],
): Uint8Array => {
  const result = new Uint8Array(vals.length * 64);
  const mask = BigInt(255)
  //const eight = BigInt(8)

  for (let i = 0; i < vals.length; i++) {
    const val = vals[i]

    //for (let j = 0; j < 16; j ++) {
      //const shift = BigInt((15 - j) * 16)
      //const a = (val >> shift) & mask
      //const b = (val >> (shift + eight)) & mask
      //result[i * 64 + (15 - j) * 4] = Number(a)
      //result[i * 64 + (15 - j) * 4 + 1] = Number(b)
    //}
 
    result[i * 64 + 60] = Number((val >> BigInt(240)) & mask)
    result[i * 64 + 61] = Number((val >> (BigInt(248))) & mask)

    result[i * 64 + 56] = Number((val >> BigInt(224)) & mask)
    result[i * 64 + 57] = Number((val >> (BigInt(232))) & mask)

    result[i * 64 + 52] = Number((val >> BigInt(208)) & mask)
    result[i * 64 + 53] = Number((val >> (BigInt(216))) & mask)

    result[i * 64 + 48] = Number((val >> BigInt(192)) & mask)
    result[i * 64 + 49] = Number((val >> (BigInt(200))) & mask)

    result[i * 64 + 44] = Number((val >> BigInt(176)) & mask)
    result[i * 64 + 45] = Number((val >> (BigInt(184))) & mask)

    result[i * 64 + 40] = Number((val >> BigInt(160)) & mask)
    result[i * 64 + 41] = Number((val >> (BigInt(168))) & mask)

    result[i * 64 + 36] = Number((val >> BigInt(144)) & mask)
    result[i * 64 + 37] = Number((val >> (BigInt(152))) & mask)

    result[i * 64 + 32] = Number((val >> BigInt(128)) & mask)
    result[i * 64 + 33] = Number((val >> (BigInt(136))) & mask)

    result[i * 64 + 28] = Number((val >> BigInt(112)) & mask)
    result[i * 64 + 29] = Number((val >> (BigInt(120))) & mask)

    result[i * 64 + 24] = Number((val >> BigInt(96)) & mask)
    result[i * 64 + 25] = Number((val >> (BigInt(104))) & mask)

    result[i * 64 + 20] = Number((val >> BigInt(80)) & mask)
    result[i * 64 + 21] = Number((val >> (BigInt(88))) & mask)

    result[i * 64 + 16] = Number((val >> BigInt(64)) & mask)
    result[i * 64 + 17] = Number((val >> (BigInt(72))) & mask)

    result[i * 64 + 12] = Number((val >> BigInt(48)) & mask)
    result[i * 64 + 13] = Number((val >> (BigInt(56))) & mask)

    result[i * 64 + 8] = Number((val >> BigInt(32)) & mask)
    result[i * 64 + 9] = Number((val >> (BigInt(40))) & mask)

    result[i * 64 + 4] = Number((val >> BigInt(16)) & mask)
    result[i * 64 + 5] = Number((val >> (BigInt(24))) & mask)

    result[i * 64 + 0] = Number((val >> BigInt(0)) & mask)
    result[i * 64 + 1] = Number((val >> (BigInt(8))) & mask)
  }

  return result;
};

export const bigint_to_u8_for_gpu = (
  val: bigint,
  num_words: number,
  word_size: number,
): Uint8Array => {
  const result = new Uint8Array(num_words * 4);
  const limbs = to_words_le(BigInt(val), num_words, word_size);
  for (let i = 0; i < limbs.length; i++) {
    const i4 = i * 4;
    result[i4] = limbs[i] & 255;
    result[i4 + 1] = limbs[i] >> 8;
  }

  return result;
};

export const gen_wgsl_limbs_code = (
  val: bigint,
  var_name: string,
  num_words: number,
  word_size: number,
): string => {
  const limbs = to_words_le(val, num_words, word_size);
  let r = "";
  for (let i = 0; i < limbs.length; i++) {
    r += `    ${var_name}.limbs[${i}]` + " = " + limbs[i].toString() + "u;\n";
  }
  return r;
};

export const gen_barrett_domb_m_limbs = (
  m: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(m, "m", num_words, word_size);
};

export const gen_p_limbs = (
  p: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(p, "p", num_words, word_size);
};

export const gen_r_limbs = (
  r: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(r, "r", num_words, word_size);
};

export const gen_d_limbs = (
  d: bigint,
  num_words: number,
  word_size: number,
): string => {
  return gen_wgsl_limbs_code(d, "d", num_words, word_size);
};

export const gen_mu_limbs = (
  p: bigint,
  num_words: number,
  word_size: number,
): string => {
  // precompute mu
  // choose x such that x is the smallest 2 ** x > s
  // https://www.nayuki.io/page/barrett-reduction-algorithm
  let x = BigInt(1);
  while (BigInt(2) ** x < p) {
    x += BigInt(1);
  }

  const mu = BigInt(4) ** x / p;
  return gen_wgsl_limbs_code(mu, "mu", num_words, word_size);
};

export const to_words_le = (
  val: bigint,
  num_words: number,
  word_size: number,
): Uint16Array => {
  const words = new Uint16Array(num_words);

  const mask = BigInt(2 ** word_size - 1);
  for (let i = 0; i < num_words; i++) {
    const idx = num_words - 1 - i;
    const shift = BigInt(idx * word_size);
    const w = (val >> shift) & mask;
    words[idx] = Number(w);
  }

  return words;
};

export const from_words_le = (
  words: Uint16Array,
  num_words: number,
  word_size: number,
): bigint => {
  assert(num_words == words.length);
  let val = BigInt(0);
  for (let i = 0; i < num_words; i++) {
    assert(words[i] < 2 ** word_size);
    assert(words[i] >= 0);
    val +=
      BigInt(2) ** BigInt((num_words - i - 1) * word_size) *
      BigInt(words[num_words - 1 - i]);
  }

  return val;
};

export const calc_num_words = (word_size: number, p_width: number): number => {
  let num_words = Math.floor(p_width / word_size);
  while (num_words * word_size < p_width) {
    num_words++;
  }
  return num_words;
};

export const compute_misc_params = (
  p: bigint,
  word_size: number,
): {
  num_words: number;
  max_terms: number;
  k: number;
  nsafe: number;
  n0: bigint;
  r: bigint;
  edwards_d: bigint;
  rinv: bigint;
  barrett_domb_m: bigint;
} => {
  const max_int_width = 32;
  assert(word_size > 0);
  const p_width = p.toString(2).length;
  const num_words = calc_num_words(word_size, p_width);
  const max_terms = num_words * 2;

  const rhs = 2 ** max_int_width;
  let k = 1;
  while (k * 2 ** (2 * word_size) <= rhs) {
    k += 1;
  }

  const nsafe = Math.floor(k / 2);

  // The Montgomery radix
  const r = BigInt(2) ** BigInt(num_words * word_size);

  // Returns triple (g, rinv, pprime)
  const egcdResult: { g: bigint; x: bigint; y: bigint } =
    bigintCryptoUtils.eGcd(r, p);
  const rinv = egcdResult.x;
  const pprime = egcdResult.y;

  if (rinv < BigInt(0)) {
    assert(((r * rinv - p * pprime) % p) + p === BigInt(1));
    assert(((r * rinv) % p) + p == BigInt(1));
    assert((p * pprime) % r == BigInt(1));
  } else {
    assert((r * rinv - p * pprime) % p === BigInt(1));
    assert((r * rinv) % p == BigInt(1));
    assert(((p * pprime) % r) + r == BigInt(1));
  }

  const neg_n_inv = r - pprime;
  const n0 = neg_n_inv % BigInt(2) ** BigInt(word_size);

  // The Barrett-Domb m value
  const z = num_words * word_size - p_width;
  const barrett_domb_m = BigInt(2 ** (2 * p_width + z)) / p;
  //m, _ = divmod(2 ** (2 * n + z), s)  # prime approximation, n + 1 bits
  const edwards_d = (EDWARDS_D * r) % p;

  return {
    num_words,
    max_terms,
    k,
    nsafe,
    n0,
    r: r % p,
    edwards_d,
    rinv,
    barrett_domb_m,
  };
};

export const genRandomFieldElement = (p: bigint): bigint => {
  // Assume that p is < 32 bytes
  const lim = BigInt(
    "0x10000000000000000000000000000000000000000000000000000000000000000",
  );
  assert(p < lim);
  const min = (lim - p) % p;

  let rand;
  while (true) {
    rand = BigInt("0x" + crypto.randomBytes(32).toString("hex"));
    if (rand >= min) {
      break;
    }
  }

  return rand % p;
};

export const are_point_arr_equal = (
  a: ExtPointType[],
  b: ExtPointType[],
): boolean => {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i++) {
    const aa = a[i].toAffine();
    const ba = b[i].toAffine();
    if (aa.x !== ba.x || aa.y !== ba.y) {
      console.log(`mismatch at ${i}`);
      return false;
    }
  }

  return true;
};
