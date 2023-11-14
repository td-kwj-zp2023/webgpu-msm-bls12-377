mod utils;

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn decompose_scalars(
    scalars: &[JsValue]
) {

    alert("Hello, decompose-scalars!");
}
