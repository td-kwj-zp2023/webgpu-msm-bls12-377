import assert from 'assert'
import crypto from 'crypto'
import * as bigintCryptoUtils from 'bigint-crypto-utils'
import { BigIntPoint } from "../reference/types"

export const points_to_u8s_for_gpu = (
    points: BigIntPoint[],
    num_words: number,
    word_size: number,
): Uint8Array => {
    const size = points.length * num_words * 4 * 4
    const result = new Uint8Array(size)

    for (let i = 0; i < points.length; i ++) {
        const x_bytes = bigint_to_u8_for_gpu(points[i].x, num_words, word_size)
        const y_bytes = bigint_to_u8_for_gpu(points[i].y, num_words, word_size)
        const t_bytes = bigint_to_u8_for_gpu(points[i].t, num_words, word_size)
        const z_bytes = bigint_to_u8_for_gpu(points[i].z, num_words, word_size)

        for (let j = 0; j < x_bytes.length; j ++) {
            result[i * x_bytes.length + j] = x_bytes[j]
            result[i * x_bytes.length + j + x_bytes.length] = y_bytes[j]
            result[i * x_bytes.length + j + x_bytes.length * 2] = t_bytes[j]
            result[i * x_bytes.length + j + x_bytes.length * 3] = z_bytes[j]
        }
    }

    return result
}

export const u8s_to_points = (
    bytes: Uint8Array,
    num_words: number,
): BigIntPoint[] => {
    // Since each limb is a u32, there are 4 u8s per limb
    const num_u8s_per_coord = num_words * 4
    const num_u8s_per_point = num_u8s_per_coord * 4

    assert(bytes.length % num_u8s_per_point === 0)
    const result: BigIntPoint[] = []
    for (let i = 0; i < bytes.length / num_u8s_per_point; i ++) {
        const x_i = i * num_u8s_per_point
        const y_i = i * num_u8s_per_point + num_u8s_per_coord
        const t_i = i * num_u8s_per_point + num_u8s_per_coord * 2
        const z_i = i * num_u8s_per_point + num_u8s_per_coord * 3

        const x_u8s = bytes.slice(x_i, x_i + num_u8s_per_coord)
        const y_u8s = bytes.slice(y_i, y_i + num_u8s_per_coord)
        const t_u8s = bytes.slice(t_i, t_i + num_u8s_per_coord)
        const z_u8s = bytes.slice(z_i, z_i + num_u8s_per_coord)

        const x = u8s_to_bigint(x_u8s)
        const y = u8s_to_bigint(y_u8s)
        const t = u8s_to_bigint(t_u8s)
        const z = u8s_to_bigint(z_u8s)

        result.push({x, y, t, z})
    }

    return result
}

export const u8s_to_bigint = (u8s: Uint8Array): bigint => {
    let val = BigInt(0)
    for (let i = 0; i < u8s.length; i ++) {
        assert(u8s[i] < 2 ** 8)
        assert(u8s[i] >= 0)
        val += (BigInt(2) ** BigInt((u8s.length - i - 1) * 8)) * BigInt(u8s[u8s.length - 1 - i])
    }

    return val
}

export const numbers_to_u8s_for_gpu = (
    vals: number[],
): Uint8Array => {
    // Expect each val to be max 32 bits
    const max = 2 ** 32
    for (const val of vals) {
        assert(val < max)
    }
    const b = new Uint32Array(vals)
    return new Uint8Array(b.buffer)
}

export const bigints_to_u8_for_gpu = (
    vals: bigint[],
    num_words: number,
    word_size: number,
): Uint8Array => {
    const size = vals.length * num_words * 4
    const result = new Uint8Array(size)

    for (let i = 0; i < vals.length; i ++) {
        const bytes = bigint_to_u8_for_gpu(vals[i], num_words, word_size)
        for (let j = 0; j < bytes.length; j ++) {
            result[i * bytes.length + j] = bytes[j]
        }
    }

    return result
}

// Mimic the functionality of
// bytemuck::cast_slice::<u32, u8>(input_bytes: &[u32]) in Rust
export const bigint_to_u8_for_gpu = (
    val: bigint,
    num_words: number,
    word_size: number,
): Uint8Array => {
    // 6672609344492007733286197091695670561995600503018092627010434561394901777994
    // 1564452267309641175680470698761959311617501419373075112580418386174943427950
    // [74, 30, 0, 0, 255, 31, 0, 0, 255, 31, 0, 0, 255, 29, 0, 0, 96, 12, 0, 0, 28, 31, 0, 0, 255, 31, 0, 0, 3, 1, 0, 0, 104, 24, 0, 0, 178, 4, 0, 0, 108, 23, 0, 0, 113, 17, 0, 0, 227, 0, 0, 0, 7, 2, 0, 0, 46, 26, 0, 0, 198, 3, 0, 0, 157, 23, 0, 0, 33, 1, 0, 0, 36, 16, 0, 0, 29, 0, 0, 0, 110, 29, 0, 0, 255, 31, 0, 0, 255, 31, 0, 0, 255, 9, 0, 0, 240, 1, 0, 0, 170, 30, 0, 0, 255, 31, 0, 0, 171, 1, 0, 0, 37, 26, 0, 0, 62, 28, 0, 0, 33, 7, 0, 0, 187, 17, 0, 0, 111, 31, 0, 0, 227, 8, 0, 0, 194, 13, 0, 0, 255, 0, 0, 0, 63, 25, 0, 0, 189, 22, 0, 0, 92, 29, 0, 0, 6, 0, 0, 0, 74, 30, 0, 0, 255, 31, 0, 0, 255, 31, 0, 0, 255, 29, 0, 0, 96, 12, 0, 0, 28, 31, 0, 0, 255, 31, 0, 0, 3, 1, 0, 0, 104, 24, 0, 0, 178, 4, 0, 0, 108, 23, 0, 0, 113, 17, 0, 0, 227, 0, 0, 0, 7, 2, 0, 0, 46, 26, 0, 0, 198, 3, 0, 0, 157, 23, 0, 0, 33, 1, 0, 0, 36, 16, 0, 0, 29, 0, 0, 0, 110, 29, 0, 0, 255, 31, 0, 0, 255, 31, 0, 0, 255, 9, 0, 0, 240, 1, 0, 0, 170, 30, 0, 0, 255, 31, 0, 0, 171, 1, 0, 0, 37, 26, 0, 0, 62, 28, 0, 0, 33, 7, 0, 0, 187, 17, 0, 0, 111, 31, 0, 0, 227, 8, 0, 0, 194, 13, 0, 0, 255, 0, 0, 0, 63, 25, 0, 0, 189, 22, 0, 0, 92, 29, 0, 0, 6, 0, 0, 0]
    const result = new Uint8Array(num_words * 4)
    const limbs = to_words_le(BigInt(val), num_words, word_size)
    for (let i = 0; i < limbs.length; i ++) {
        const b = new Uint8Array(Uint16Array.from([limbs[i]]).buffer)
        result[(i * 4)] = b[0]
        result[(i * 4) + 1] = b[1]
        result[(i * 4) + 2] = 0
        result[(i * 4) + 3] = 0
    }

    return result
}

export const gen_p_limbs = (
    p: bigint,
    num_words: number,
    word_size: number,
): string => {
    const p_limbs = to_words_le(p, num_words, word_size)
    let r = ''
    for (let i = 0; i < p_limbs.length; i ++) {
        r += `    p.limbs[${i}]` + ' \= ' + p_limbs[i].toString() + 'u;\n'
        //r += `    p.limbs[${i}]=${p_limbs[i].toString()}u;\n`
    }
    return r
}

export const to_words_le = (val: bigint, num_words: number, word_size: number): Uint16Array => {
    const words = new Uint16Array(num_words)

    // max value per limb (exclusive)
    const max_limb_size = BigInt(2 ** word_size)

    let v = val
    let i = 0
    while (v > 0) {
        const limb = v % max_limb_size
        words[i] = Number(limb)
        v = v / max_limb_size
        i ++
    }

    return words
}

export const from_words_le = (words: Uint16Array, num_words: number, word_size: number): bigint => {
    assert(num_words == words.length)
    let val = BigInt(0)
    for (let i = 0; i < num_words; i ++) {
        assert(words[i] < 2 ** word_size)
        assert(words[i] >= 0)
        val += (BigInt(2) ** BigInt((num_words - i - 1) * word_size)) * BigInt(words[num_words - 1 - i])
    }

    return val
}

export const calc_num_words = (word_size: number, p_width: number): number => {
    let num_words = Math.floor(p_width / word_size)
    while (num_words * word_size < p_width) {
        num_words ++
    }
    return num_words
}

export const compute_misc_params = (
    p: bigint,
    word_size: number,
): {
        num_words: number,
        max_terms: number,
        k: number,
        nsafe: number,
        n0: bigint
        r: bigint
} => {
    const max_int_width = 32
    assert(word_size > 0)
    const p_width = p.toString(2).length
    const num_words = calc_num_words(word_size, p_width)
    const max_terms = num_words * 2

    const rhs = 2 ** max_int_width
    let k = 1
    while (k * (2 ** (2 * word_size)) <= rhs) {
        k += 1
    }

    const nsafe = Math.floor(k / 2)

    // The Montgomery radix
    const r = BigInt(2) ** BigInt(num_words * word_size)

    // Returns triple (g, rinv, pprime)
    const egcdResult: {g: bigint, x: bigint, y: bigint} = bigintCryptoUtils.eGcd(r, p);
    const rinv = egcdResult.x
    const pprime = egcdResult.y

    if (rinv < BigInt(0)) {
        assert((r * rinv - p * pprime) % p + p === BigInt(1))
        assert((r * rinv) % p + p == BigInt(1))
        assert((p * pprime) % r == BigInt(1))
    } else {
        assert((r * rinv - p * pprime) % p === BigInt(1))
        assert((r * rinv) % p == BigInt(1))
        assert((p * pprime) % r + r == BigInt(1))
    }

    const neg_n_inv = r - pprime
    const n0 = neg_n_inv % (BigInt(2) ** BigInt(word_size))

    return { num_words, max_terms, k, nsafe, n0, r: r % p }
}

export const genRandomFieldElement = (p: bigint): bigint => {
    // Assume that p is < 32 bytes
    const lim = BigInt('0x10000000000000000000000000000000000000000000000000000000000000000')
    assert(p < lim)
    const min = (lim - p) % p

    let rand
    while (true) {
        rand = BigInt('0x' + crypto.randomBytes(32).toString('hex'))
        if (rand >= min) {
            break
        }
    }

    return rand % p
}

// Copied from src/reference/webgpu/utils.ts
// TODO: rewrite them?

export interface gpuU32Inputs {
  u32Inputs: Uint32Array;
  individualInputSize: number;
}

export const bigIntsToU16Array = (beBigInts: bigint[]): Uint16Array => {
  const intsAs16s = beBigInts.map(bigInt => bigIntToU16Array(bigInt));
  const u16Array = new Uint16Array(beBigInts.length * 16);
  intsAs16s.forEach((intAs16, index) => {u16Array.set(intAs16, index * 16)});
  return u16Array;
}

export const bigIntToU16Array = (beBigInt: bigint): Uint16Array => {
  const numBits = 256;
  const bitsPerElement = 16;
  const numElements = numBits / bitsPerElement;
  const u16Array = new Uint16Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u16Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u16Array;
};

export const flattenU32s = (u32Arrays: Uint32Array[]): Uint32Array => {
  const flattenedU32s = new Uint32Array(u32Arrays.length * u32Arrays[0].length);
  u32Arrays.forEach((u32Array, index) => {
    flattenedU32s.set(u32Array, index * u32Array.length);
  });
  return flattenedU32s;
};

// assume bigints are big endian 256-bit integers
export const bigIntsToU32Array = (beBigInts: bigint[]): Uint32Array => {
  const intsAs32s = beBigInts.map(bigInt => bigIntToU32Array(bigInt));
  const u32Array = new Uint32Array(beBigInts.length * 8);
  intsAs32s.forEach((intAs32, index) => {u32Array.set(intAs32, index * 8)});
  return u32Array;
};

export const bigIntToU32Array = (beBigInt: bigint): Uint32Array => {
  const numBits = 256;
  const bitsPerElement = 32;
  const numElements = numBits / bitsPerElement;
  const u32Array = new Uint32Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u32Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u32Array;
};

export const u32ArrayToBigInts = (u32Array: Uint32Array): bigint[] => {
  const bigInts = [];
  const chunkSize = 8;
  const bitsPerElement = 32;

  for (let i = 0; i < u32Array.length; i += chunkSize) {
    let bigInt = BigInt(0);
    for (let j = 0; j < chunkSize; j++) {
      if (i + j >= u32Array.length) break; // Avoid out-of-bounds access
      const u32 = BigInt(u32Array[i + j]);
      bigInt |= (u32 << (BigInt(chunkSize - 1 - j) * BigInt(bitsPerElement)));
    }
    bigInts.push(bigInt);
  }

  return bigInts;
};

export const convertBigIntsToWasmFields = (bigInts: bigint[]): string[] => {
  return bigInts.map(bigInt => bigInt.toString() + 'field');
};

export const stripFieldSuffix = (field: string): string => {
  return field.slice(0, field.length - 5);
};

export const stripGroupSuffix = (group: string): string => {
  return group.slice(0, group.length - 5);
};

export const chunkArray = (inputsArray: gpuU32Inputs[], batchSize: number): gpuU32Inputs[][] => {
  let index = 0;
  const chunkedArray: gpuU32Inputs[][] = [];
  const firstInputLength = inputsArray[0].u32Inputs.length / inputsArray[0].individualInputSize;

  while (index < firstInputLength) {
      const newIndex = index + batchSize;
      const tempArray: gpuU32Inputs[] = [];
      inputsArray.forEach(bufferData => {
        const chunkedGpuU32Inputs = bufferData.u32Inputs.slice(index * bufferData.individualInputSize, newIndex * bufferData.individualInputSize);
        tempArray.push({
          u32Inputs: chunkedGpuU32Inputs,
          individualInputSize: bufferData.individualInputSize
        });
      });
      index = newIndex;
      chunkedArray.push(tempArray);
  }

  return chunkedArray;
};

export function concatUint32Arrays(array1: Uint32Array, array2: Uint32Array): Uint32Array {
  // Create a new Uint32Array with a length equal to the sum of the lengths of array1 and array2
  const result = new Uint32Array(array1.length + array2.length);

  // Copy the elements from array1 into the new array
  result.set(array1, 0);

  // Copy the elements from array2 into the new array, starting at the index after the last element of array1
  result.set(array2, array1.length);

  return result;
}