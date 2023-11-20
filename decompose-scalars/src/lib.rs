mod utils;

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use js_sys::{BigInt, Array, ArrayBuffer};
use std::ops::Shr;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

pub fn to_words_le(
    val: BigInt,
    num_words: usize,
    word_size: usize,
) -> Vec<u16> {
    let words: Vec<u16> = Vec::with_capacity(num_words);
    return words;
}

#[wasm_bindgen]
pub struct Result {
    result: Vec<u16>
}

#[wasm_bindgen]
impl Result {
    pub fn get_result(&self) -> Vec<u16> {
        self.result.clone()
    }
}

pub fn extract_word_from_bytes_le(
    input: &Vec<u8>,
    word_idx: usize,
    word_size: usize,
) -> u16 {
    let start_byte_idx = input.len() - 1 - ((word_idx * word_size + word_size) / 8);
    //log(serde_json::to_string(&[input.len(), word_idx, word_size, start_byte_idx]).unwrap().as_str());
    let end_byte_idx = input.len() - 1 - ((word_idx * word_size) / 8);
    let start_byte_offset = (word_idx * word_size + word_size) % 8;
    let end_byte_offset = (word_idx * word_size) % 8;

    let mut sum = 0u16;
    for i in start_byte_idx..(end_byte_idx + 1) {
        if i == start_byte_idx {
            let mask = 2u16.pow(start_byte_offset as u32) - 1;
            sum += input[i] as u16 & mask;
        } else if i == end_byte_idx {
            sum = sum << (8 - end_byte_offset);
            sum += input[i] as u16 >> end_byte_offset;
        } else {
            sum = sum << 8;
            sum += input[i] as u16;
        }
    }
    return sum;
}

#[wasm_bindgen]
pub fn decompose_scalars(
    scalars: JsValue,
    num_words: usize,
    word_size: usize,
) -> Result {
    let scalars_array = Array::from(&scalars);
    let bigints: Vec<BigInt> = scalars_array.iter().map(|x| BigInt::new(&x).unwrap()).collect();

    let mut r: Vec<Vec<u16>> = Vec::with_capacity(bigints.len());

    let max_hex_len: usize = (((num_words * word_size) as f64 / 8f64).ceil() as usize) * 2;
    for bigint in &bigints {
        let mut b = bigint.to_string(16).unwrap().as_string().unwrap();
        while b.len() < max_hex_len {
            b = format!("0{}", b);
        }

        let bytes = hex::decode(b).unwrap();

        let mut words = vec![0u16; num_words];
        for i in 0..num_words {
            words[i] = extract_word_from_bytes_le(&bytes, i, word_size); 
        }
        r.push(words);
    }


    let mut result: Vec<u16> = Vec::with_capacity(bigints.len() * num_words);
    for j in 0..num_words {
        for i in 0..bigints.len() {
            result.push(r[i][j]);
        }
    }
    Result {
        result
    }
}
