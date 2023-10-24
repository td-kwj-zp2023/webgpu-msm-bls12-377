struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

struct Point {
  x: BigInt,
  y: BigInt,
  t: BigInt,
  z: BigInt
}

struct BigIntMediumWide {
    limbs: array<u32, {{ num_words_plus_one }}>
}