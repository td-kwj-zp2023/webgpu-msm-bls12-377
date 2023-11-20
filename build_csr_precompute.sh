#!/usr/bin/env bash

cd "$(dirname "$0")"
cd csr-precompute
wasm-pack build --release
