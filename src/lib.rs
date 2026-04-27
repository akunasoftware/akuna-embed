//! Simple text embedding models built with Burn.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_embed::{EmbeddingModel, TextEmbedding, TextEmbeddingOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = TextEmbedding::new(TextEmbeddingOptions {
//!         model: EmbeddingModel::MiniLmL12,
//!         ..Default::default()
//!     })
//!     .await?;
//!
//!     let single = model.embed("Hello world")?;
//!     assert!(!single.is_empty());
//!
//!     let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;
//!     assert_eq!(batch.len(), 2);
//!
//!     Ok(())
//! }
//! ```

mod bert;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};
use burn_wgpu::{Wgpu, WgpuDevice};

use crate::bert::{
    BertEmbeddingModel, BertEmbeddingVariant, EmbeddingInputKind,
    load_pretrained_bert_embedding,
};

pub type DefaultBackend = Wgpu;
pub type DefaultDevice = WgpuDevice;
const DEFAULT_BATCH_SIZE: usize = 32;

/// Supported embedding model checkpoints.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EmbeddingModel {
    MiniLmL6,
    #[default]
    MiniLmL12,
    BgeSmallEnV15,
    BgeBaseEnV15,
}

impl From<EmbeddingModel> for BertEmbeddingVariant {
    fn from(value: EmbeddingModel) -> Self {
        match value {
            EmbeddingModel::MiniLmL6 => BertEmbeddingVariant::MiniLmL6,
            EmbeddingModel::MiniLmL12 => BertEmbeddingVariant::MiniLmL12,
            EmbeddingModel::BgeSmallEnV15 => {
                BertEmbeddingVariant::BgeSmallEnV15
            }
            EmbeddingModel::BgeBaseEnV15 => BertEmbeddingVariant::BgeBaseEnV15,
        }
    }
}

/// Options for [`TextEmbedding`].
#[derive(Debug, Clone, Default)]
pub struct TextEmbeddingOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// Minimal text embedding interface inspired by `fastembed-rs`.
#[derive(Debug)]
pub struct TextEmbedding<B: Backend = DefaultBackend> {
    model: BertEmbeddingModel<B>,
    device: B::Device,
}

impl TextEmbedding<DefaultBackend> {
    /// Loads a MiniLM text embedding model onto the default WGPU device.
    pub async fn new(options: TextEmbeddingOptions) -> Result<Self> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> TextEmbedding<B>
where
    B: Backend,
{
    /// Loads a MiniLM text embedding model onto the provided device.
    pub async fn new_with_device(
        device: &B::Device,
        options: TextEmbeddingOptions,
    ) -> Result<Self> {
        let model = load_pretrained_bert_embedding(
            device,
            options.model.into(),
            options.cache_dir,
        )
        .await?;

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Embeds a single document and returns one embedding vector.
    pub fn embed(&self, document: impl AsRef<str>) -> Result<Vec<f32>> {
        let document = document.as_ref();
        let documents = [document];
        let mut embeddings = self.embed_batch(documents.as_slice(), None)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input document")
    }

    /// Embeds a search query using any model-specific retrieval prompt.
    ///
    /// Some retrieval models train queries and documents with different text
    /// prefixes. Use this when the text is the thing being searched for.
    /// Use [`TextEmbedding::embed`] when the text is the content being indexed.
    pub fn embed_query(&self, query: impl AsRef<str>) -> Result<Vec<f32>> {
        let query = query.as_ref();
        let queries = [query];
        let mut embeddings =
            self.embed_query_batch(queries.as_slice(), None)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input query")
    }

    /// Embeds documents in batches and returns one vector per input string.
    pub fn embed_batch<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_kind(
            documents,
            batch_size,
            EmbeddingInputKind::Document,
        )
    }

    /// Embeds search queries in batches using model-specific retrieval prompts.
    ///
    /// See [`TextEmbedding::embed_query`] for when query embeddings differ from
    /// document embeddings.
    pub fn embed_query_batch<S: AsRef<str>>(
        &self,
        queries: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_kind(
            queries,
            batch_size,
            EmbeddingInputKind::Query,
        )
    }

    fn embed_batch_with_kind<S: AsRef<str>>(
        &self,
        inputs: &[S],
        batch_size: Option<usize>,
        input_kind: EmbeddingInputKind,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch_size_or_default(inputs.len(), batch_size)?;

        let mut embeddings = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(batch_size) {
            let batch_inputs =
                batch.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            let batch_embeddings =
                self.model.encode(&batch_inputs, input_kind, &self.device)?;
            embeddings.extend(tensor_to_rows(batch_embeddings)?);
        }

        Ok(embeddings)
    }

    /// Returns the loaded embedding checkpoint.
    pub fn model(&self) -> EmbeddingModel {
        match self.model.variant {
            BertEmbeddingVariant::MiniLmL6 => EmbeddingModel::MiniLmL6,
            BertEmbeddingVariant::MiniLmL12 => EmbeddingModel::MiniLmL12,
            BertEmbeddingVariant::BgeSmallEnV15 => {
                EmbeddingModel::BgeSmallEnV15
            }
            BertEmbeddingVariant::BgeBaseEnV15 => EmbeddingModel::BgeBaseEnV15,
        }
    }
}

fn batch_size_or_default(
    document_count: usize,
    batch_size: Option<usize>,
) -> Result<usize> {
    let batch_size =
        batch_size.unwrap_or(document_count.min(DEFAULT_BATCH_SIZE));
    if batch_size == 0 {
        bail!("batch size must be greater than zero");
    }

    Ok(batch_size)
}

fn tensor_to_rows<B: Backend>(
    embeddings: Tensor<B, 2>,
) -> Result<Vec<Vec<f32>>> {
    let [row_count, column_count] = embeddings.dims();
    let data = embeddings.into_data().convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to read embedding output tensor")?;

    Ok(values
        .chunks(column_count)
        .take(row_count)
        .map(|row| row.to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_wgpu::{Wgpu, WgpuDevice};
    use std::sync::OnceLock;
    use tokio::sync::Mutex;

    static LIVE_MODEL_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[test]
    fn api_model_mapping_converts_all_public_variants() {
        assert_eq!(
            BertEmbeddingVariant::from(EmbeddingModel::MiniLmL6),
            BertEmbeddingVariant::MiniLmL6
        );
        assert_eq!(
            BertEmbeddingVariant::from(EmbeddingModel::MiniLmL12),
            BertEmbeddingVariant::MiniLmL12
        );
        assert_eq!(
            BertEmbeddingVariant::from(EmbeddingModel::BgeSmallEnV15),
            BertEmbeddingVariant::BgeSmallEnV15
        );
        assert_eq!(
            BertEmbeddingVariant::from(EmbeddingModel::BgeBaseEnV15),
            BertEmbeddingVariant::BgeBaseEnV15
        );
    }

    #[test]
    fn api_model_metadata_returns_bge_repo_ids() {
        assert_eq!(
            BertEmbeddingVariant::BgeSmallEnV15.repo_id(),
            "BAAI/bge-small-en-v1.5"
        );
        assert_eq!(
            BertEmbeddingVariant::BgeBaseEnV15.repo_id(),
            "BAAI/bge-base-en-v1.5"
        );
    }

    #[test]
    fn api_options_default_uses_minilm_l12() {
        assert_eq!(
            TextEmbeddingOptions::default().model,
            EmbeddingModel::MiniLmL12
        );
    }

    #[tokio::test]
    async fn model_bge_small_embed_returns_document_and_query_vectors() {
        let _guard = live_model_test_lock().lock().await;
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model: EmbeddingModel::BgeSmallEnV15,
            ..Default::default()
        })
        .await
        .expect("model should load");

        let document = model
            .embed("Hello world")
            .expect("document embed should work");
        let query = model
            .embed_query("Hello world")
            .expect("query embed should work");

        assert_eq!(document.len(), 384);
        assert_eq!(query.len(), 384);
    }

    #[tokio::test]
    async fn model_minilm_l6_backend_supports_i32_indices() {
        let _guard = live_model_test_lock().lock().await;
        let device = WgpuDevice::default();
        let model = TextEmbedding::<Wgpu<f32, i32>>::new_with_device(
            &device,
            TextEmbeddingOptions {
                model: EmbeddingModel::MiniLmL6,
                cache_dir: None,
            },
        )
        .await
        .expect("model should load");

        let single = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!single.is_empty());
    }

    #[tokio::test]
    async fn model_minilm_l6_embed_returns_vectors() {
        let _guard = live_model_test_lock().lock().await;
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model: EmbeddingModel::MiniLmL6,
            ..Default::default()
        })
        .await
        .expect("model should load");

        let single = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!single.is_empty());

        let batch = model
            .embed_batch(&["Hello world", "Rust embeddings"], None)
            .expect("batch embed should work");
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|embedding| !embedding.is_empty()));
    }

    #[tokio::test]
    async fn parity_bge_base_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeBaseEnV15,
            "BAAI/bge-base-en-v1.5",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_bge_base_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeBaseEnV15,
            "BAAI/bge-base-en-v1.5",
            ReferenceInputKind::Query,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_bge_small_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeSmallEnV15,
            "BAAI/bge-small-en-v1.5",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_bge_small_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeSmallEnV15,
            "BAAI/bge-small-en-v1.5",
            ReferenceInputKind::Query,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_minilm_l12_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::MiniLmL12,
            "sentence-transformers/all-MiniLM-L12-v2",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_minilm_l12_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::MiniLmL12,
            "sentence-transformers/all-MiniLM-L12-v2",
            ReferenceInputKind::Query,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_minilm_l6_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::MiniLmL6,
            "sentence-transformers/all-MiniLM-L6-v2",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_minilm_l6_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::MiniLmL6,
            "sentence-transformers/all-MiniLM-L6-v2",
            ReferenceInputKind::Query,
        )
        .await;
    }

    #[test]
    fn util_batch_size_default_caps_large_batches() {
        let batch_size = batch_size_or_default(128, None)
            .expect("default batch size should work");
        assert_eq!(batch_size, DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn util_batch_size_default_uses_document_count_when_small() {
        let batch_size = batch_size_or_default(4, None)
            .expect("default batch size should work");
        assert_eq!(batch_size, 4);
    }

    #[test]
    fn util_batch_size_validate_rejects_zero() {
        let error = batch_size_or_default(1, Some(0))
            .expect_err("zero batch size should fail");
        assert!(
            error
                .to_string()
                .contains("batch size must be greater than zero")
        );
    }

    #[test]
    fn util_tensor_rows_extract_returns_rows() {
        let device = WgpuDevice::default();
        let embeddings = Tensor::<Wgpu<f32, i64>, 2>::from_floats(
            [[1.0, 2.0], [3.0, 4.0]],
            &device,
        );

        let rows = tensor_to_rows(embeddings).expect("rows should extract");
        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[derive(Debug, Clone, Copy)]
    enum ReferenceInputKind {
        Document,
        Query,
    }

    impl ReferenceInputKind {
        fn as_str(self) -> &'static str {
            match self {
                Self::Document => "document",
                Self::Query => "query",
            }
        }
    }

    async fn assert_model_matches_sentence_transformers(
        model: EmbeddingModel,
        reference_model: &str,
        input_kind: ReferenceInputKind,
    ) {
        let _guard = live_model_test_lock().lock().await;
        let texts =
            vec!["Hello world".to_string(), "Rust embeddings".to_string()];
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model,
            ..Default::default()
        })
        .await
        .expect("model should load");
        let actual = match input_kind {
            ReferenceInputKind::Document => model
                .embed_batch(&texts, Some(2))
                .expect("Burn document embeddings should work"),
            ReferenceInputKind::Query => model
                .embed_query_batch(&texts, Some(2))
                .expect("Burn query embeddings should work"),
        };
        let expected =
            reference_embeddings(reference_model, input_kind.as_str(), &texts)
                .expect("reference embeddings should work");

        assert_embedding_batches_close(&actual, &expected, 1e-3, 0.999);
    }

    fn live_model_test_lock() -> &'static Mutex<()> {
        LIVE_MODEL_TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn reference_embeddings(
        model: &str,
        kind: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("uv")
            .args([
                "run",
                "scripts/reference_embeddings.py",
                "--model",
                model,
                "--kind",
                kind,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn uv reference embedding script")?;

        let mut stdin = child
            .stdin
            .take()
            .context("failed to open reference script stdin")?;
        let input = serde_json::to_vec(texts)
            .context("failed to serialize reference input")?;
        stdin
            .write_all(&input)
            .context("failed to write reference input")?;
        drop(stdin);

        let output = child
            .wait_with_output()
            .context("failed to wait for reference script")?;
        if !output.status.success() {
            bail!(
                "reference script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        serde_json::from_slice(&output.stdout)
            .context("failed to parse reference embeddings")
    }

    fn assert_embedding_batches_close(
        actual: &[Vec<f32>],
        expected: &[Vec<f32>],
        tolerance: f32,
        min_cosine_similarity: f32,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.len(), expected.len());
            let max_delta = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_delta <= tolerance,
                "max embedding delta {max_delta} exceeded tolerance {tolerance}"
            );
            let cosine_similarity = cosine_similarity(actual, expected);
            assert!(
                cosine_similarity >= min_cosine_similarity,
                "cosine similarity {cosine_similarity} fell below {min_cosine_similarity}"
            );
        }
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let dot_product = left
            .iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let left_norm =
            left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm =
            right.iter().map(|value| value * value).sum::<f32>().sqrt();

        dot_product / (left_norm * right_norm)
    }
}
