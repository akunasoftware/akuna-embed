#!/usr/bin/env bash

set -euo pipefail

cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
cargo check --all-features
cargo nextest run --all-features --no-capture -j 1
cargo test --doc --all-features
cargo deny check --allow=no-license-field
