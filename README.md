[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/v/akuna-embed?style=for-the-badge)](https://crates.io/crates/akuna-embed)
[![Docs.rs](https://img.shields.io/docsrs/akuna-embed?style=for-the-badge)](https://docs.rs/akuna-embed)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna-embed?style=for-the-badge)](https://github.com/akunasoftware/akuna-embed/commits/main)
[![CI](https://img.shields.io/github/actions/workflow/status/akunasoftware/akuna-embed/ci.yml?label=ci&style=for-the-badge)](https://github.com/akunasoftware/akuna-embed/actions/workflows/ci.yml)

# akuna-embed

Simple pure-rust text embedding models built on [Burn](https://github.com/tracel-ai/burn).

- No external model runtimes, and native hardware acceleration.
- Super simple interface, give text and get embeddings.
- Optionally, specify a model, and/or a Burn backend for different hardware execution.

## Usage

```rust
use akuna_embed::TextEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(Default::default()).await?;

    let embedding = model.embed("Hello world")?;
    println!("Embedding has {} numbers", embedding.len());

    Ok(())
}
```

## Embed Many Texts

```rust
use akuna_embed::TextEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(Default::default()).await?;

    let embeddings = model.embed_batch(&[
        "Hello world",
        "Rust embeddings",
        "Semantic search",
    ], None)?;

    println!("Created {} embeddings", embeddings.len());

    Ok(())
}
```

## Search Usage

When building search, embed stored content with `embed` or `embed_batch`.

Embed user search text with `embed_query` or `embed_query_batch`.
These methods follow `sentence-transformers` defaults and do not add hidden
model-specific prompts.

If a model card recommends a prompt, pass it explicitly with
`embed_query_with_prompt` or `embed_query_batch_with_prompt`.

```rust
use akuna_embed::TextEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(Default::default()).await?;

    let document = model.embed("Burn is a deep learning framework for Rust")?;
    let query = model.embed_query("Rust machine learning")?;

    assert_eq!(document.len(), query.len());

    Ok(())
}
```

```rust,no_run
use akuna_embed::TextEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(Default::default()).await?;
    let prompt = "Represent this sentence for searching relevant passages: ";

    let query = model.embed_query_with_prompt(
        "Rust machine learning",
        Some(prompt),
    )?;

    assert!(!query.is_empty());

    Ok(())
}
```

## Choose A Model

`EmbeddingModel::MiniLmL12` is the default.

Available models:

- `EmbeddingModel::MiniLmL12`
- `EmbeddingModel::MiniLmL6`
- `EmbeddingModel::BgeSmallEnV15`
- `EmbeddingModel::BgeBaseEnV15`
- `EmbeddingModel::BgeLargeEnV15`
- `EmbeddingModel::BgeM3`
- `EmbeddingModel::AllMpnetBaseV2`

`EmbeddingModel::BgeM3` exposes the dense embedding output only.
Sparse and multi-vector BGE-M3 outputs are not part of this crate's simple
`Vec<f32>` embedding API.

```rust,no_run
use akuna_embed::{EmbeddingModel, TextEmbedding, TextEmbeddingOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(TextEmbeddingOptions {
        model: EmbeddingModel::BgeSmallEnV15,
        ..Default::default()
    })
    .await?;

    let embedding = model.embed("Hello world")?;
    assert!(!embedding.is_empty());

    Ok(())
}
```

## Development

This project uses a Nix development shell.

If you use `nix-direnv`, it should activate automatically.
To enter it manually:

```sh
nix develop
```

Run all checks with:

```sh
./scripts/check.sh
```

Run tests only with:

```sh
cargo nextest run
```

Tests compare Rust output with Python `sentence-transformers` reference
embeddings through `uv run scripts/reference_embeddings.py`.

BGE-M3 parity tests are ignored by default because the model is large enough to
exhaust WGPU memory in the full live suite.
Run them explicitly with:

```sh
cargo test parity_bge_m3_ -- --ignored --nocapture --test-threads=1
```
