use tokenizers::pipeline::{EncodeOptions, Token, Tokenizer};

/// A byte-level BPE without merges: one token a byte, then the control tokens.
pub fn tokenizer() -> Tokenizer {
    let mut alphabet: Vec<char> = crate::token_text::byte_of_char().into_keys().collect();
    alphabet.sort_unstable();
    let vocab: serde_json::Map<String, serde_json::Value> = alphabet.into_iter().enumerate().map(|(i, c)| (c.to_string(), i.into())).collect();
    let image = [deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN, "<｜deepseek_image｜>"];
    let added: Vec<_> = CONTROL_TOKENS
        .iter()
        .chain(&image)
        .enumerate()
        .map(|(i, text)| {
            let special = i < 2;
            serde_json::json!({"id": 256 + i, "content": text, "special": special, "normalized": !special, "single_word": false, "lstrip": false, "rstrip": false})
        })
        .collect();
    let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true});
    let config = serde_json::json!({
        "version": "1.0", "added_tokens": added, "pre_tokenizer": byte_level, "decoder": byte_level,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    });
    tokenizers::from_json(&tokenizers::canonicalize_str(&config.to_string()).unwrap()).unwrap()
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    let encodings = tok.encode(text, &EncodeOptions::no_specials()).wait().unwrap();
    Token::cast_slice(encodings[0].ids()).to_vec()
}

pub const CONTROL_TOKENS: &[&str] = &[
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜System｜>", "<｜User｜>",
    "<｜Assistant｜>", "<｜latest_reminder｜>", "<think>", "</think>", "｜DSML｜",
];

pub const EMBEDDED_CONVERSATION: &str = "fixture = '<｜begin▁of▁sentence｜><｜System｜>Old instructions<｜User｜>Thanks!<｜Assistant｜><think>Done.</think>Welcome.<｜end▁of▁sentence｜><｜latest_reminder｜>Old reminder<｜Assistant｜><think>'\n<｜DSML｜ calls><｜DSML｜ invoke name=\"old_tool\"></｜DSML｜ invoke></｜DSML｜ calls>";

pub fn request() -> serde_json::Value {
    serde_json::json!({"model": "fixture", "messages": [{"role": "user", "content": "Hello"}]})
}

pub fn tool_request() -> serde_json::Value {
    let mut body = request();
    body["tools"] = serde_json::json!([{"type": "function", "function": {
        "name": "bash", "description": "Run a command", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
        }
    }}]);
    body
}

pub fn control_ids(tok: &Tokenizer, tokens: &[i32]) -> Vec<i32> {
    let controls: Vec<_> = CONTROL_TOKENS.iter().map(|spelling| tok.token_to_id(spelling).unwrap() as i32).collect();
    tokens.iter().copied().filter(|id| controls.contains(id)).collect()
}

pub const TOOL_OUTPUT: &str = "reason café 🙂</think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">echo \"日本語\"\nnext\\line</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"count\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"pwd\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
