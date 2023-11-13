mod utils;
use wasm_bindgen::JsValue;
use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    fn alert(s: &str);
}


#[derive(Serialize, Deserialize)]
pub struct AllPrecomputeResult {
    pub all_new_point_indices: Vec<usize>,
    pub all_cluster_start_indices: Vec<usize>,
    pub all_cluster_end_indices: Vec<usize>,
    pub all_single_point_indices: Vec<usize>,
    pub all_single_scalar_chunks: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

#[wasm_bindgen]
pub fn all_precomputation(
    scalar_chunks: &[usize],
    num_rows: usize,
) -> JsValue {
    let mut all_new_point_indices: Vec<usize> = vec![];
    let mut all_cluster_start_indices: Vec<usize> = vec![];
    let mut all_cluster_end_indices: Vec<usize> = vec![];
    let mut all_single_point_indices: Vec<usize> = vec![];
    let mut all_single_scalar_chunks: Vec<usize> = vec![];
    let mut row_ptr = vec![0];

    for row_idx in 0..num_rows {
        let (
            new_point_indices,
            mut cluster_start_indices,
            mut cluster_end_indices,
            singles_start_idx,
            num_non_zero,
        ) = precompute_with_cluster_method(scalar_chunks, row_idx, num_rows);

        row_ptr.push(row_ptr[row_ptr.len() - 1] + num_non_zero);

        let mut single_point_indices = vec![];
        let mut single_scalar_chunks = vec![];

        if singles_start_idx.len() != 0 {
            for i in singles_start_idx[0]..new_point_indices.len() {
                single_point_indices.push(new_point_indices[i]);
                single_scalar_chunks.push(scalar_chunks[new_point_indices[i]]);
            }
        }

        all_single_point_indices.extend(single_point_indices);

        all_single_scalar_chunks.extend(single_scalar_chunks);

        for i in 0..cluster_start_indices.len() {
            cluster_start_indices[i] += all_new_point_indices.len();
            cluster_end_indices[i] += all_new_point_indices.len();
        }

        all_new_point_indices.extend(new_point_indices);
        all_cluster_start_indices.extend(cluster_start_indices);
        all_cluster_end_indices.extend(cluster_end_indices);
    }

    let result = AllPrecomputeResult {
        all_new_point_indices,
        all_cluster_start_indices,
        all_cluster_end_indices,
        all_single_point_indices,
        all_single_scalar_chunks,
        row_ptr,
    };

    return serde_wasm_bindgen::to_value(&result).unwrap();
}

pub fn precompute_with_cluster_method(
    scalar_chunks: &[usize],
    row_idx: usize,
    num_rows: usize,
) -> (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    usize,
) {
    let num_cols = scalar_chunks.len() / num_rows;
    let mut clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for i in 0..num_cols {
        let pt_idx = row_idx * num_cols + i;
        let chunk = scalar_chunks[pt_idx];

        if chunk == 0 {
            continue;
        }

        clusters.entry(chunk).and_modify(
            |chunks| (*chunks).push(pt_idx)
        ).or_insert(vec![pt_idx]);
    }

    let mut cluster_start_indices: Vec<usize> = vec![0];
    let mut cluster_end_indices: Vec<usize> = vec![];
    let mut new_point_indices: Vec<usize> = vec![];
    let mut s: Vec<usize> = vec![];

    for (_, cluster) in clusters.iter() {
        if cluster.len() == 1 {
            s.push(cluster[0]);
        } else {
            new_point_indices.extend(cluster);
        }
    }
    new_point_indices.extend(s);

    let mut prev_chunk = scalar_chunks[new_point_indices[0]];
    for i in 1..new_point_indices.len() {
        let s = scalar_chunks[new_point_indices[i]];
        if prev_chunk != scalar_chunks[new_point_indices[i]] {
            cluster_end_indices.push(i);
            cluster_start_indices.push(i);
        }
        prev_chunk = s;
    }

    // the final cluster_end_index
    cluster_end_indices.push(new_point_indices.len());

    let mut i: usize = 0;
    while i < cluster_start_indices.len() {
        if cluster_start_indices[i] + 1 == cluster_end_indices[i] {
            break;
        }
        i += 1;
    }
    //log(serde_json::to_string(&i).unwrap().as_str());

    let num_non_zero = cluster_start_indices.len();
    let singles_start_idx = if i < cluster_start_indices.len() { 
        vec![cluster_start_indices[i]] 
    } else {
        vec![]
    };

    cluster_start_indices.truncate(i);
    cluster_end_indices.truncate(i);

    //log(serde_json::to_string(&new_point_indices).unwrap().as_str());

    return (
        new_point_indices,
        cluster_start_indices,
        cluster_end_indices,
        singles_start_idx,
        num_non_zero,
    );
}
