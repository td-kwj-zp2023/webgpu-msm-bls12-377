//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;
use csr_precompute::{
    all_precomputation,
    AllPrecomputeResult,
};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn pass() {
    let num_rows = 2;
    let scalar_chunk = vec![4, 4, 4, 3, 3, 3, 3, 0];
    //let scalar_chunk = vec![3, 3, 2, 1, 2, 1, 4, 4];
    let result: AllPrecomputeResult = serde_wasm_bindgen::from_value(all_precomputation(scalar_chunk.as_slice(), num_rows)).unwrap();
}
