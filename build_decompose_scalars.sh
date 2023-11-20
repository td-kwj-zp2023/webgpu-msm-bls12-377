#!/usr/bin/env bash

cd "$(dirname "$0")"
cd decompose-scalars
wasm-pack build --release
