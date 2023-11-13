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
    let scalar_chunks = vec![4, 4, 4, 3, 3, 3, 3, 0];
    let result: AllPrecomputeResult = serde_wasm_bindgen::from_value(all_precomputation(scalar_chunks.as_slice(), num_rows)).unwrap();
    println!("{}", result.all_new_point_indices.len());
    println!("{:?}", result.all_new_point_indices);
    println!("{:?}", result.all_cluster_start_indices);
    println!("{:?}", result.all_cluster_end_indices);
    println!("{:?}", result.all_single_point_indices);
    println!("{:?}", result.all_single_scalar_chunks);
    println!("{:?}", result.row_ptr);
}
