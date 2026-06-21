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
mod mpnet;
mod xlm_roberta;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};
use burn_wgpu::{Wgpu, WgpuDevice};

use crate::bert::{
    BertEmbeddingModel, BertEmbeddingVariant, EmbeddingInputKind,
    load_pretrained_bert_embedding,
};
use crate::mpnet::{
    MpnetEmbeddingModel, MpnetEmbeddingVariant, load_pretrained_mpnet_embedding,
};
use crate::xlm_roberta::{
    XlmRobertaEmbeddingModel, XlmRobertaEmbeddingVariant,
    load_pretrained_xlm_roberta_embedding,
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
    BgeLargeEnV15,
    AllMpnetBaseV2,
    /// BGE-M3 dense embeddings only.
    ///
    /// Sparse and multi-vector outputs are separate retrieval concerns and are
    /// not exposed through this `Vec<f32>` dense embedding API.
    BgeM3,
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
            EmbeddingModel::BgeLargeEnV15 => {
                BertEmbeddingVariant::BgeLargeEnV15
            }
            EmbeddingModel::AllMpnetBaseV2 | EmbeddingModel::BgeM3 => {
                unreachable!("non-BERT models use their own loaders")
            }
        }
    }
}

impl From<EmbeddingModel> for MpnetEmbeddingVariant {
    fn from(value: EmbeddingModel) -> Self {
        match value {
            EmbeddingModel::AllMpnetBaseV2 => {
                MpnetEmbeddingVariant::AllMpnetBaseV2
            }
            EmbeddingModel::MiniLmL6
            | EmbeddingModel::MiniLmL12
            | EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15
            | EmbeddingModel::BgeM3 => {
                unreachable!("non-MPNet models use their own loaders")
            }
        }
    }
}

impl From<EmbeddingModel> for XlmRobertaEmbeddingVariant {
    fn from(value: EmbeddingModel) -> Self {
        match value {
            EmbeddingModel::BgeM3 => XlmRobertaEmbeddingVariant::BgeM3,
            EmbeddingModel::MiniLmL6
            | EmbeddingModel::MiniLmL12
            | EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15
            | EmbeddingModel::AllMpnetBaseV2 => {
                unreachable!("non-XLM-RoBERTa models use their own loaders")
            }
        }
    }
}

#[derive(Debug)]
enum LoadedEmbeddingModel<B: Backend> {
    Bert(BertEmbeddingModel<B>),
    Mpnet(MpnetEmbeddingModel<B>),
    XlmRoberta(XlmRobertaEmbeddingModel<B>),
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
    model: LoadedEmbeddingModel<B>,
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
        let model = match options.model {
            EmbeddingModel::MiniLmL6
            | EmbeddingModel::MiniLmL12
            | EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15 => LoadedEmbeddingModel::Bert(
                load_pretrained_bert_embedding(
                    device,
                    options.model.into(),
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::AllMpnetBaseV2 => LoadedEmbeddingModel::Mpnet(
                load_pretrained_mpnet_embedding(
                    device,
                    options.model.into(),
                    options.cache_dir,
                )
                .await?,
            ),
            EmbeddingModel::BgeM3 => LoadedEmbeddingModel::XlmRoberta(
                load_pretrained_xlm_roberta_embedding(
                    device,
                    options.model.into(),
                    options.cache_dir,
                )
                .await?,
            ),
        };

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Embeds a single document and returns one embedding vector.
    pub fn embed(&self, document: impl AsRef<str>) -> Result<Vec<f32>> {
        self.embed_with_prompt(document, None)
    }

    /// Embeds a single document with an optional input prompt.
    pub fn embed_with_prompt(
        &self,
        document: impl AsRef<str>,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        let document = document.as_ref();
        let documents = [document];
        let mut embeddings =
            self.embed_batch_with_prompt(documents.as_slice(), None, prompt)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input document")
    }

    /// Embeds a search query using reference default behavior.
    pub fn embed_query(&self, query: impl AsRef<str>) -> Result<Vec<f32>> {
        self.embed_query_with_prompt(query, None)
    }

    /// Embeds a search query with an optional input prompt.
    pub fn embed_query_with_prompt(
        &self,
        query: impl AsRef<str>,
        prompt: Option<&str>,
    ) -> Result<Vec<f32>> {
        let query = query.as_ref();
        let queries = [query];
        let mut embeddings = self.embed_query_batch_with_prompt(
            queries.as_slice(),
            None,
            prompt,
        )?;
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
        self.embed_batch_with_prompt(documents, batch_size, None)
    }

    /// Embeds documents with an optional input prompt.
    pub fn embed_batch_with_prompt<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_kind(
            documents,
            batch_size,
            EmbeddingInputKind::Document,
            prompt,
        )
    }

    /// Embeds search queries in batches using reference default behavior.
    pub fn embed_query_batch<S: AsRef<str>>(
        &self,
        queries: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_query_batch_with_prompt(queries, batch_size, None)
    }

    /// Embeds search queries in batches with an optional input prompt.
    pub fn embed_query_batch_with_prompt<S: AsRef<str>>(
        &self,
        queries: &[S],
        batch_size: Option<usize>,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_kind(
            queries,
            batch_size,
            EmbeddingInputKind::Query,
            prompt,
        )
    }

    fn embed_batch_with_kind<S: AsRef<str>>(
        &self,
        inputs: &[S],
        batch_size: Option<usize>,
        input_kind: EmbeddingInputKind,
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch_size_or_default(inputs.len(), batch_size)?;

        let mut embeddings = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(batch_size) {
            let batch_inputs =
                batch.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            let batch_embeddings = match &self.model {
                LoadedEmbeddingModel::Bert(model) => model.encode(
                    &batch_inputs,
                    input_kind,
                    prompt,
                    &self.device,
                )?,
                LoadedEmbeddingModel::Mpnet(model) => model.encode(
                    &batch_inputs,
                    input_kind,
                    prompt,
                    &self.device,
                )?,
                LoadedEmbeddingModel::XlmRoberta(model) => model.encode(
                    &batch_inputs,
                    input_kind,
                    prompt,
                    &self.device,
                )?,
            };
            embeddings.extend(tensor_to_rows(batch_embeddings)?);
        }

        Ok(embeddings)
    }

    /// Returns the loaded embedding checkpoint.
    pub fn model(&self) -> EmbeddingModel {
        match &self.model {
            LoadedEmbeddingModel::Bert(model) => match model.variant {
                BertEmbeddingVariant::MiniLmL6 => EmbeddingModel::MiniLmL6,
                BertEmbeddingVariant::MiniLmL12 => EmbeddingModel::MiniLmL12,
                BertEmbeddingVariant::BgeSmallEnV15 => {
                    EmbeddingModel::BgeSmallEnV15
                }
                BertEmbeddingVariant::BgeBaseEnV15 => {
                    EmbeddingModel::BgeBaseEnV15
                }
                BertEmbeddingVariant::BgeLargeEnV15 => {
                    EmbeddingModel::BgeLargeEnV15
                }
            },
            LoadedEmbeddingModel::Mpnet(model) => match model.variant {
                MpnetEmbeddingVariant::AllMpnetBaseV2 => {
                    EmbeddingModel::AllMpnetBaseV2
                }
            },
            LoadedEmbeddingModel::XlmRoberta(model) => match model.variant {
                XlmRobertaEmbeddingVariant::BgeM3 => EmbeddingModel::BgeM3,
            },
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
    const BGE_QUERY_PROMPT: &str =
        "Represent this sentence for searching relevant passages: ";

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
        assert_eq!(
            BertEmbeddingVariant::from(EmbeddingModel::BgeLargeEnV15),
            BertEmbeddingVariant::BgeLargeEnV15
        );
        assert_eq!(
            MpnetEmbeddingVariant::from(EmbeddingModel::AllMpnetBaseV2),
            MpnetEmbeddingVariant::AllMpnetBaseV2
        );
        assert_eq!(
            XlmRobertaEmbeddingVariant::from(EmbeddingModel::BgeM3),
            XlmRobertaEmbeddingVariant::BgeM3
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
        assert_eq!(
            BertEmbeddingVariant::BgeLargeEnV15.repo_id(),
            "BAAI/bge-large-en-v1.5"
        );
        assert_eq!(
            MpnetEmbeddingVariant::AllMpnetBaseV2.repo_id(),
            "sentence-transformers/all-mpnet-base-v2"
        );
        assert_eq!(XlmRobertaEmbeddingVariant::BgeM3.repo_id(), "BAAI/bge-m3");
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
    async fn parity_bge_large_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeLargeEnV15,
            "BAAI/bge-large-en-v1.5",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_bge_large_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeLargeEnV15,
            "BAAI/bge-large-en-v1.5",
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
    async fn parity_bge_small_query_with_prompt_matches_sentence_transformers()
    {
        assert_model_matches_sentence_transformers_with_prompt(
            EmbeddingModel::BgeSmallEnV15,
            "BAAI/bge-small-en-v1.5",
            ReferenceInputKind::Query,
            BGE_QUERY_PROMPT,
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

    #[tokio::test]
    async fn parity_mpnet_base_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::AllMpnetBaseV2,
            "sentence-transformers/all-mpnet-base-v2",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    async fn parity_mpnet_base_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::AllMpnetBaseV2,
            "sentence-transformers/all-mpnet-base-v2",
            ReferenceInputKind::Query,
        )
        .await;
    }

    #[tokio::test]
    #[ignore = "BGE-M3 is too large for the default WGPU parity suite"]
    async fn parity_bge_m3_document_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeM3,
            "BAAI/bge-m3",
            ReferenceInputKind::Document,
        )
        .await;
    }

    #[tokio::test]
    #[ignore = "BGE-M3 is too large for the default WGPU parity suite"]
    async fn parity_bge_m3_query_matches_sentence_transformers() {
        assert_model_matches_sentence_transformers(
            EmbeddingModel::BgeM3,
            "BAAI/bge-m3",
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
        assert_model_matches_sentence_transformers_for_texts(
            model,
            reference_model,
            input_kind,
            parity_texts(),
        )
        .await;
        assert_model_matches_sentence_transformers_for_texts(
            model,
            reference_model,
            input_kind,
            long_parity_texts(),
        )
        .await;
    }

    async fn assert_model_matches_sentence_transformers_with_prompt(
        model: EmbeddingModel,
        reference_model: &str,
        input_kind: ReferenceInputKind,
        prompt: &str,
    ) {
        let _guard = live_model_test_lock().lock().await;
        assert_model_matches_sentence_transformers_for_texts_with_prompt(
            model,
            reference_model,
            input_kind,
            parity_texts(),
            Some(prompt),
        )
        .await;
    }

    async fn assert_model_matches_sentence_transformers_for_texts(
        model: EmbeddingModel,
        reference_model: &str,
        input_kind: ReferenceInputKind,
        texts: Vec<String>,
    ) {
        assert_model_matches_sentence_transformers_for_texts_with_prompt(
            model,
            reference_model,
            input_kind,
            texts,
            None,
        )
        .await;
    }

    async fn assert_model_matches_sentence_transformers_for_texts_with_prompt(
        model: EmbeddingModel,
        reference_model: &str,
        input_kind: ReferenceInputKind,
        texts: Vec<String>,
        prompt: Option<&str>,
    ) {
        let model = TextEmbedding::new(TextEmbeddingOptions {
            model,
            ..Default::default()
        })
        .await
        .expect("model should load");
        let actual = match input_kind {
            ReferenceInputKind::Document => model
                .embed_batch_with_prompt(&texts, Some(2), prompt)
                .expect("Burn document embeddings should work"),
            ReferenceInputKind::Query => model
                .embed_query_batch_with_prompt(&texts, Some(2), prompt)
                .expect("Burn query embeddings should work"),
        };
        let expected = reference_embeddings(
            reference_model,
            input_kind.as_str(),
            &texts,
            prompt,
        )
        .expect("reference embeddings should work");

        assert_embedding_batches_close(
            &actual,
            &expected,
            &texts,
            model.model(),
            input_kind,
            max_delta_tolerance(model.model()),
            0.999,
        );
    }

    fn max_delta_tolerance(model: EmbeddingModel) -> f32 {
        match model {
            EmbeddingModel::BgeM3 => 1e-2,
            EmbeddingModel::MiniLmL6
            | EmbeddingModel::MiniLmL12
            | EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15
            | EmbeddingModel::AllMpnetBaseV2 => 1e-3,
        }
    }

    fn parity_texts() -> Vec<String> {
        vec![
            "Hello world".to_string(),
            "Rust embeddings".to_string(),
            "Semantic search: fast, accurate, and simple.".to_string(),
            "  padded input with leading and trailing spaces  ".to_string(),
            "Numbers 12345, symbols !?., and mixed CASE.".to_string(),
            "emoji rocket and unicode cafe".to_string(),
        ]
        .into_iter()
        .chain(multilingual_parity_texts())
        .collect()
    }

    fn multilingual_parity_texts() -> Vec<String> {
        include_str!("../test-corpus/test-multilingual.txt")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect()
    }

    fn long_parity_texts() -> Vec<String> {
        let sentence = "Burn embeddings should match sentence-transformers even when tokenizer truncation is required. ";
        vec![sentence.repeat(128)]
    }

    fn live_model_test_lock() -> &'static Mutex<()> {
        LIVE_MODEL_TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn reference_embeddings(
        model: &str,
        kind: &str,
        texts: &[String],
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut args = vec![
            "run",
            "scripts/reference_embeddings.py",
            "--model",
            model,
            "--kind",
            kind,
        ];
        if let Some(prompt) = prompt {
            args.extend(["--prompt", prompt]);
        }

        let mut child = Command::new("uv")
            .args(args)
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
        texts: &[String],
        model: EmbeddingModel,
        input_kind: ReferenceInputKind,
        tolerance: f32,
        min_cosine_similarity: f32,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in
            actual.iter().zip(expected).enumerate()
        {
            assert_eq!(
                actual.len(),
                expected.len(),
                "embedding width mismatch for {model:?} {input_kind:?} input {index}: {:?}",
                texts.get(index)
            );
            let max_delta = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_delta <= tolerance,
                "max embedding delta {max_delta} exceeded tolerance {tolerance} for {model:?} {input_kind:?} input {index}: {:?}",
                texts.get(index)
            );
            let cosine_similarity = cosine_similarity(actual, expected);
            assert!(
                cosine_similarity >= min_cosine_similarity,
                "cosine similarity {cosine_similarity} fell below {min_cosine_similarity} for {model:?} {input_kind:?} input {index}: {:?}",
                texts.get(index)
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
