use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig,
    transformer::{
        TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
    },
};
use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use hf_hub::api::tokio::ApiBuilder;
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

const MINILM_L6_REPO_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MINILM_L12_REPO_ID: &str = "sentence-transformers/all-MiniLM-L12-v2";
const BGE_SMALL_EN_V15_REPO_ID: &str = "BAAI/bge-small-en-v1.5";
const BGE_BASE_EN_V15_REPO_ID: &str = "BAAI/bge-base-en-v1.5";
const BGE_LARGE_EN_V15_REPO_ID: &str = "BAAI/bge-large-en-v1.5";
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum BertEmbeddingVariant {
    MiniLmL6,
    #[default]
    MiniLmL12,
    BgeSmallEnV15,
    BgeBaseEnV15,
    BgeLargeEnV15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingInputKind {
    Query,
    Document,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoolingStrategy {
    Mean,
    Cls,
}

struct BertEmbeddingMetadata {
    repo_id: &'static str,
    pooling_strategy: PoolingStrategy,
}

impl BertEmbeddingVariant {
    pub fn repo_id(self) -> &'static str {
        self.metadata().repo_id
    }

    fn pooling_strategy(self) -> PoolingStrategy {
        self.metadata().pooling_strategy
    }

    fn metadata(self) -> BertEmbeddingMetadata {
        match self {
            Self::MiniLmL6 => BertEmbeddingMetadata {
                repo_id: MINILM_L6_REPO_ID,
                pooling_strategy: PoolingStrategy::Mean,
            },
            Self::MiniLmL12 => BertEmbeddingMetadata {
                repo_id: MINILM_L12_REPO_ID,
                pooling_strategy: PoolingStrategy::Mean,
            },
            Self::BgeSmallEnV15 => BertEmbeddingMetadata {
                repo_id: BGE_SMALL_EN_V15_REPO_ID,
                pooling_strategy: PoolingStrategy::Cls,
            },
            Self::BgeBaseEnV15 => BertEmbeddingMetadata {
                repo_id: BGE_BASE_EN_V15_REPO_ID,
                pooling_strategy: PoolingStrategy::Cls,
            },
            Self::BgeLargeEnV15 => BertEmbeddingMetadata {
                repo_id: BGE_LARGE_EN_V15_REPO_ID,
                pooling_strategy: PoolingStrategy::Cls,
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct HfModelFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) weights_path: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
    pub(crate) sentence_bert_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct BertConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SentenceBertConfig {
    max_seq_length: Option<usize>,
}

impl SentenceBertConfig {
    pub(crate) fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).with_context(|| {
            format!(
                "failed to read sentence-transformers config at {}",
                path.display()
            )
        })?;

        serde_json::from_str(&content).with_context(|| {
            format!(
                "failed to parse sentence-transformers config at {}",
                path.display()
            )
        })
    }

    pub(crate) fn max_seq_length(&self) -> Option<usize> {
        self.max_seq_length
    }
}

#[derive(Debug)]
struct BertOutput<B: Backend> {
    hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct BertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
struct BertModel<B: Backend> {
    embeddings: BertEmbeddings<B>,
    encoder: TransformerEncoder<B>,
}

#[derive(Debug)]
pub(crate) struct BertEmbeddingModel<B: Backend> {
    model: BertModel<B>,
    tokenizer: Tokenizer,
    pub(crate) variant: BertEmbeddingVariant,
}

impl BertConfig {
    pub fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).with_context(|| {
            format!("failed to read embedding config at {}", path.display())
        })?;

        serde_json::from_str(&content).with_context(|| {
            format!("failed to parse embedding config at {}", path.display())
        })
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embeddings = BertEmbeddings::new(self, device);
        let encoder = TransformerEncoderConfig::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_hidden_layers,
        )
        .with_dropout(0.0)
        .with_layer_norm_eps(self.layer_norm_eps)
        .with_norm_first(false)
        .with_quiet_softmax(false)
        .with_initializer(KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: false,
        })
        .init(device);

        BertModel {
            embeddings,
            encoder,
        }
    }
}

impl<B: Backend> BertEmbeddings<B> {
    fn new(config: &BertConfig, device: &B::Device) -> Self {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device);
        let position_embeddings = EmbeddingConfig::new(
            config.max_position_embeddings,
            config.hidden_size,
        )
        .init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(config.type_vocab_size, config.hidden_size)
                .init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(0.0).init();

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let word_embeddings = self.word_embeddings.forward(input_ids);

        let position_ids =
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
                .reshape([1, seq_len])
                .expand([batch_size, seq_len]);
        let position_embeddings =
            self.position_embeddings.forward(position_ids);

        let token_type_ids = token_type_ids.unwrap_or_else(|| {
            Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device)
        });
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids);

        let embeddings =
            word_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> BertModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> BertOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);
        let encoder_input =
            TransformerEncoderInput::new(embeddings).mask_pad(mask_pad);
        let hidden_states = self.encoder.forward(encoder_input);

        BertOutput { hidden_states }
    }
}

impl<B> BertEmbeddingModel<B>
where
    B: Backend,
{
    pub(crate) fn encode(
        &self,
        sentences: &[&str],
        _input_kind: EmbeddingInputKind,
        prompt: Option<&str>,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let prompted_sentences = prompt_sentences(sentences, prompt);
        let prompted_sentence_refs = prompted_sentences
            .iter()
            .map(Cow::as_ref)
            .collect::<Vec<_>>();
        let (input_ids, attention_mask) = tokenize_batch::<B>(
            &self.tokenizer,
            &prompted_sentence_refs,
            device,
        )?;
        let output =
            self.model.forward(input_ids, attention_mask.clone(), None);

        let embeddings = match self.variant.pooling_strategy() {
            PoolingStrategy::Mean => {
                mean_pooling(output.hidden_states, attention_mask)
            }
            PoolingStrategy::Cls => cls_pooling(output.hidden_states),
        };

        Ok(normalize_l2(embeddings))
    }
}

pub(crate) fn prompt_sentences<'a>(
    sentences: &[&'a str],
    prompt: Option<&str>,
) -> Vec<Cow<'a, str>> {
    // SentenceTransformers strips input strings before tokenization.
    sentences
        .iter()
        .map(|sentence| match prompt {
            Some(prompt) => Cow::Owned(format!("{prompt}{}", sentence.trim())),
            None => Cow::Borrowed(sentence.trim()),
        })
        .collect()
}

pub(crate) async fn load_pretrained_bert_embedding<B>(
    device: &B::Device,
    variant: BertEmbeddingVariant,
    cache_dir: Option<PathBuf>,
) -> Result<BertEmbeddingModel<B>>
where
    B: Backend,
{
    let files = download_hf_model(variant.repo_id(), cache_dir).await?;
    let config = BertConfig::load_from_hf(&files.config_path)?;
    let mut model = config.init(device);
    load_pretrained_weights(&mut model, &files.weights_path)?;
    let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!(
                "failed to load embedding tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?;
    let max_length = sentence_transformers_max_length(
        files.sentence_bert_config_path.as_deref(),
    )?
    .unwrap_or(config.max_position_embeddings)
    .min(config.max_position_embeddings);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(BertEmbeddingModel {
        model,
        tokenizer,
        variant,
    })
}

pub(crate) async fn download_hf_model(
    repo_id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles> {
    download_hf_model_with_weights(repo_id, "model.safetensors", cache_dir)
        .await
}

pub(crate) async fn download_hf_model_with_weights(
    repo_id: &str,
    weights_file: &str,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }

    let api = builder
        .build()
        .context("failed to initialize Hugging Face API for embedding model")?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo.get("config.json").await.with_context(|| {
        format!("failed to fetch embedding config for {repo_id}")
    })?;
    let weights_path = repo.get(weights_file).await.with_context(|| {
        format!("failed to fetch embedding weights for {repo_id}")
    })?;
    let tokenizer_path =
        repo.get("tokenizer.json").await.with_context(|| {
            format!("failed to fetch embedding tokenizer for {repo_id}")
        })?;
    let sentence_bert_config_path =
        repo.get("sentence_bert_config.json").await.ok();

    Ok(HfModelFiles {
        config_path,
        weights_path,
        tokenizer_path,
        sentence_bert_config_path,
    })
}

pub(crate) fn sentence_transformers_max_length(
    path: Option<&Path>,
) -> Result<Option<usize>> {
    path.map(SentenceBertConfig::load_from_hf)
        .transpose()
        .map(|config| config.and_then(|config| config.max_seq_length()))
}

fn load_pretrained_weights<B: Backend>(
    model: &mut BertModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let key_mappings = vec![
        ("^bert\\.(.+)", "$1"),
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        ("attention\\.self\\.query", "mha.query"),
        ("attention\\.self\\.key", "mha.key"),
        ("attention\\.self\\.value", "mha.value"),
        ("attention\\.output\\.dense", "mha.output"),
        ("attention\\.output\\.LayerNorm", "norm_1"),
        ("intermediate\\.dense", "pwff.linear_inner"),
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.pwff.linear_outer"),
        ("(layers\\.[0-9]+)\\.output\\.LayerNorm", "$1.norm_2"),
        ("embeddings\\.LayerNorm", "embeddings.layer_norm"),
    ];

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create embedding weight remapper")?;
    let mut store = SafetensorsStore::from_file(checkpoint_path.as_ref())
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load embedding weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

pub(crate) fn tokenize_batch<B: Backend>(
    tokenizer: &Tokenizer,
    sentences: &[&str],
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2>)> {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize input batch")?;

    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1);

    let batch_size = sentences.len();
    let mut input_ids = vec![0i32; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for (token_index, token_id) in encoding.get_ids().iter().enumerate() {
            input_ids[batch_index * max_len + token_index] = *token_id as i32;
            attention_mask[batch_index * max_len + token_index] =
                encoding.get_attention_mask()[token_index] as f32;
        }
    }

    let input_ids =
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]);
    let attention_mask =
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]);

    Ok((input_ids, attention_mask))
}

pub(crate) fn mean_pooling<B: Backend>(
    hidden_states: Tensor<B, 3>,
    attention_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    let mask_expanded = attention_mask
        .clone()
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);
    let sum_hidden = (hidden_states * mask_expanded)
        .sum_dim(1)
        .reshape([batch_size, hidden_size]);
    let token_counts = attention_mask
        .sum_dim(1)
        .reshape([batch_size, 1])
        .expand([batch_size, hidden_size])
        .clamp_min(1e-9);

    sum_hidden / token_counts
}

fn cls_pooling<B: Backend>(hidden_states: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    let device = hidden_states.device();
    let mut mask = vec![0.0f32; batch_size * seq_len];
    for batch_index in 0..batch_size {
        mask[batch_index * seq_len] = 1.0;
    }

    let mask = Tensor::<B, 1>::from_floats(mask.as_slice(), &device)
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);

    (hidden_states * mask)
        .sum_dim(1)
        .reshape([batch_size, hidden_size])
}

pub(crate) fn normalize_l2<B: Backend>(
    embeddings: Tensor<B, 2>,
) -> Tensor<B, 2> {
    use burn::tensor::linalg::{Norm, vector_normalize};

    vector_normalize(embeddings, Norm::L2, 1, 1e-12)
}
