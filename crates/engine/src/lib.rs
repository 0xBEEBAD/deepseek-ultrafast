mod decode_cache;
pub mod engram;
pub mod fp4;
pub mod kvstore;
pub mod model;
pub mod ops;
pub mod sampling;
pub mod weights;
pub mod vision;
mod vision_attention;
mod weight_staging;

pub fn load_tokenizer(ckpt: &std::path::Path) -> anyhow::Result<tokenizers::pipeline::Tokenizer> {
    let json = tokenizers::canonicalize_file(ckpt.join("tokenizer.json"))?;
    tokenizers::from_json(&json).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
}

/// One past the largest token id. `get_vocab_size(true)` counts a special token twice when the model's vocabulary has it too.
pub fn vocab_size(tok: &tokenizers::pipeline::Tokenizer) -> usize {
    tok.get_vocab(true).values().max().map_or(0, |&id| id as usize + 1)
}

pub fn load_config(ckpt: &std::path::Path) -> anyhow::Result<deepseek_ultrafast_model::config::Config> {
    let mut cfg = deepseek_ultrafast_model::config::Config::load(&ckpt.join("inference/config.json"))?;
    if let Some(b) = deepseek_ultrafast_cuda::profile::dspark_block() {
        cfg.dspark_block_size = b;
    }
    Ok(cfg)
}
