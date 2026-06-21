use std::path::{Path, PathBuf};

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
use burn_store::{KeyRemapper, ModuleSnapshot, PytorchStore};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::bert::{
    EmbeddingInputKind, download_hf_model_with_weights, normalize_l2,
    prompt_sentences, sentence_transformers_max_length,
};

const BGE_M3_REPO_ID: &str = "BAAI/bge-m3";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum XlmRobertaEmbeddingVariant {
    #[default]
    BgeM3,
}

impl XlmRobertaEmbeddingVariant {
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::BgeM3 => BGE_M3_REPO_ID,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct XlmRobertaConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: usize,
    #[serde(default = "default_pad_token_id")]
    pad_token_id: i32,
}

#[derive(Debug)]
struct XlmRobertaOutput<B: Backend> {
    hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct XlmRobertaEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
struct XlmRobertaModel<B: Backend> {
    embeddings: XlmRobertaEmbeddings<B>,
    encoder: TransformerEncoder<B>,
}

#[derive(Debug)]
pub(crate) struct XlmRobertaEmbeddingModel<B: Backend> {
    model: XlmRobertaModel<B>,
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: i32,
    pub(crate) variant: XlmRobertaEmbeddingVariant,
}

impl XlmRobertaConfig {
    fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).with_context(|| {
            format!("failed to read embedding config at {}", path.display())
        })?;

        serde_json::from_str(&content).with_context(|| {
            format!("failed to parse embedding config at {}", path.display())
        })
    }

    fn init<B: Backend>(&self, device: &B::Device) -> XlmRobertaModel<B> {
        let embeddings = XlmRobertaEmbeddings::new(self, device);
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

        XlmRobertaModel {
            embeddings,
            encoder,
        }
    }
}

impl<B: Backend> XlmRobertaEmbeddings<B> {
    fn new(config: &XlmRobertaConfig, device: &B::Device) -> Self {
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
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let word_embeddings = self.word_embeddings.forward(input_ids);
        let [batch_size, seq_len] = position_ids.dims();
        let device = position_ids.device();
        let position_embeddings =
            self.position_embeddings.forward(position_ids);
        let token_type_ids =
            Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device);
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids);
        let embeddings =
            word_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);

        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> XlmRobertaModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        position_ids: Tensor<B, 2, Int>,
    ) -> XlmRobertaOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, position_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);
        let encoder_input =
            TransformerEncoderInput::new(embeddings).mask_pad(mask_pad);
        let hidden_states = self.encoder.forward(encoder_input);

        XlmRobertaOutput { hidden_states }
    }
}

impl<B> XlmRobertaEmbeddingModel<B>
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
            .map(std::borrow::Cow::as_ref)
            .collect::<Vec<_>>();
        let (input_ids, attention_mask, position_ids) = tokenize_batch::<B>(
            &self.tokenizer,
            &prompted_sentence_refs,
            self.max_length,
            self.pad_token_id,
            device,
        )?;
        let output =
            self.model.forward(input_ids, attention_mask, position_ids);
        let embeddings = cls_pooling(output.hidden_states);

        Ok(normalize_l2(embeddings))
    }
}

pub(crate) async fn load_pretrained_xlm_roberta_embedding<B>(
    device: &B::Device,
    variant: XlmRobertaEmbeddingVariant,
    cache_dir: Option<PathBuf>,
) -> Result<XlmRobertaEmbeddingModel<B>>
where
    B: Backend,
{
    let files = download_hf_model_with_weights(
        variant.repo_id(),
        "pytorch_model.bin",
        cache_dir,
    )
    .await?;
    let config = XlmRobertaConfig::load_from_hf(&files.config_path)?;
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
    .min(config.max_position_embeddings.saturating_sub(2));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(XlmRobertaEmbeddingModel {
        model,
        tokenizer,
        max_length,
        pad_token_id: config.pad_token_id,
        variant,
    })
}

fn load_pretrained_weights<B: Backend>(
    model: &mut XlmRobertaModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let key_mappings = vec![
        ("^roberta\\.(.+)", "$1"),
        ("^xlm_roberta\\.(.+)", "$1"),
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
    let mut store = PytorchStore::from_file(checkpoint_path.as_ref())
        .map_indices_contiguous(false)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load embedding weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

fn tokenize_batch<B: Backend>(
    tokenizer: &Tokenizer,
    sentences: &[&str],
    max_length: usize,
    pad_token_id: i32,
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2, Int>)> {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize input batch")?;

    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1)
        .min(max_length);
    let batch_size = sentences.len();
    let mut input_ids = vec![pad_token_id; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];
    let mut position_ids = vec![pad_token_id; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        let mut position_id = pad_token_id + 1;
        for token_index in 0..encoding.get_ids().len().min(max_len) {
            let offset = batch_index * max_len + token_index;
            let token_id = encoding.get_ids()[token_index] as i32;
            let mask = encoding.get_attention_mask()[token_index] as f32;
            input_ids[offset] = token_id;
            attention_mask[offset] = mask;
            if mask > 0.0 {
                position_ids[offset] = position_id;
                position_id += 1;
            }
        }
    }

    let input_ids =
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]);
    let attention_mask =
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]);
    let position_ids =
        Tensor::<B, 1, Int>::from_ints(position_ids.as_slice(), device)
            .reshape([batch_size, max_len]);

    Ok((input_ids, attention_mask, position_ids))
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

fn default_pad_token_id() -> i32 {
    1
}

fn default_type_vocab_size() -> usize {
    1
}
