//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn pass() {
    let mut hex_str = String::from("e55121e109c9c728149a54395afba711cc8ed4afc28d72bea7404963dcfcf74");
    if hex_str.len() % 2 == 1 {
        hex_str = format!("0{}", hex_str);
    }
    let mut bytes = [0u8; 32];
    let _ = hex::decode_to_slice(hex_str, &mut bytes as &mut [u8]).unwrap();
    assert_eq!(1 + 1, 2);
}
