use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::pipeline::{EncodeOptions, Segment, SpecialSegmentIterator, Token, Tokenizer};

const CACHED_BYTES: usize = 256;
const PARALLEL_BYTES: usize = 64 << 10;
const CAPACITY_BYTES: usize = 256 << 20;

struct Entry {
    ids: Arc<[u32]>,
    used: u64,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<String, Entry>,
    bytes: usize,
    clock: u64,
}

impl Cache {
    fn insert(&mut self, text: &str, ids: Arc<[u32]>) {
        let size = text.len() + 4 * ids.len();
        if size > CAPACITY_BYTES / 4 || self.entries.contains_key(text) {
            return;
        }
        self.bytes += size;
        self.entries.insert(text.to_owned(), Entry { ids, used: self.clock });
        if self.bytes > CAPACITY_BYTES {
            let mut ages: Vec<u64> = self.entries.values().map(|e| e.used).collect();
            let half = ages.len() / 2;
            let cutoff = *ages.select_nth_unstable(half).1;
            self.entries.retain(|_, e| e.used > cutoff);
            self.bytes = self.entries.iter().map(|(k, e)| k.len() + 4 * e.ids.len()).sum();
        }
    }
}

struct Registered {
    tok: Arc<Tokenizer>,
    cache: Mutex<Cache>,
}

static REGISTERED: OnceLock<Registered> = OnceLock::new();

pub fn register(tok: Arc<Tokenizer>) {
    let _ = REGISTERED.set(Registered { tok, cache: Mutex::default() });
}

fn cache_for(tok: &Tokenizer) -> Option<&'static Mutex<Cache>> {
    REGISTERED.get().filter(|r| std::ptr::eq(Arc::as_ptr(&r.tok), tok)).map(|r| &r.cache)
}

enum Piece<'a> {
    Special(u32),
    Text(&'a str),
}

fn pieces<'a>(tok: &Tokenizer, text: &'a str) -> Vec<Piece<'a>> {
    let added = tok.get_added_vocabulary();
    // The second search finds the tokens declared on normalized text. It reads the raw text, which is only right without normalizers.
    let search_normalized = tok.get_normalizers().is_empty();
    let mut pieces = Vec::new();
    for raw in SpecialSegmentIterator::new(text, added, false, false) {
        match raw {
            Segment::SpecialToken(id) => pieces.push(Piece::Special(id)),
            Segment::Text { text, .. } if search_normalized => {
                pieces.extend(SpecialSegmentIterator::new(text, added, true, false).map(|segment| match segment {
                    Segment::SpecialToken(id) => Piece::Special(id),
                    Segment::Text { text, .. } => Piece::Text(text),
                }));
            }
            Segment::Text { text, .. } => pieces.push(Piece::Text(text)),
        }
    }
    pieces
}

pub fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let pieces = pieces(tok, text);
    let cache = cache_for(tok);
    let mut done: Vec<Option<Arc<[u32]>>> = vec![None; pieces.len()];
    if let Some(cache) = cache {
        let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
        c.clock += 1;
        let now = c.clock;
        for (i, piece) in pieces.iter().enumerate() {
            if let Piece::Text(s) = piece
                && s.len() >= CACHED_BYTES
                && let Some(e) = c.entries.get_mut(*s)
            {
                e.used = now;
                done[i] = Some(e.ids.clone());
            }
        }
    }
    let misses: Vec<(usize, &str)> = pieces
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Piece::Text(s) if done[i].is_none() => Some((i, *s)),
            _ => None,
        })
        .collect();
    let options = EncodeOptions::no_specials();
    let run = |&(_, s): &(usize, &str)| -> Result<Arc<[u32]>> {
        let mut ids = Vec::new();
        tok.encode_into(s, &options, &mut ids).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Token::cast_slice(&ids).into())
    };
    let parallel = misses.len() > 1 && misses.iter().map(|(_, s)| s.len()).sum::<usize>() >= PARALLEL_BYTES;
    let encoded: Vec<Arc<[u32]>> = if parallel {
        misses.par_iter().map(run).collect::<Result<_>>()?
    } else {
        misses.iter().map(run).collect::<Result<_>>()?
    };
    if let Some(cache) = cache {
        let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
        for ((_, s), ids) in misses.iter().zip(&encoded) {
            if s.len() >= CACHED_BYTES {
                c.insert(s, ids.clone());
            }
        }
    }
    for ((i, _), ids) in misses.iter().zip(encoded) {
        done[*i] = Some(ids);
    }
    let mut out = Vec::new();
    for (piece, ids) in pieces.iter().zip(&done) {
        match (piece, ids) {
            (Piece::Special(id), _) => out.push(*id),
            (Piece::Text(_), Some(ids)) => out.extend_from_slice(ids),
            (Piece::Text(_), None) => unreachable!(),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(tok: &Tokenizer, text: &str) -> Vec<u32> {
        crate::test_support::encode(tok, text)
    }

    #[test]
    fn matches_whole_encode_on_the_test_tokenizer() {
        let tok = crate::test_support::tokenizer();
        for text in ["", "hello world", "a\n\nb", "x".repeat(70_000).as_str()] {
            assert_eq!(encode(&tok, text).unwrap(), reference(&tok, text), "{text:.40}");
        }
    }

    #[test]
    fn matches_whole_encode_on_the_checkpoint_tokenizer() {
        let Ok(dir) = std::env::var("DSUF_TEST_CKPT") else { return };
        let tok = Arc::new(deepseek_ultrafast_engine::load_tokenizer(std::path::Path::new(&dir)).unwrap());
        register(tok.clone());
        assert!(cache_for(&tok).is_some());
        let mut specials: Vec<String> = tok.get_added_vocabulary().get_added_tokens_decoder().into_values().map(|t| t.content).collect();
        specials.sort();
        let body = "fn main() {\n    println!(\"héllo 世界 🙂 12345\");\n}\n\tTabs  and   spaces.\n".repeat(40);
        let mut texts = vec![body.clone(), String::new(), " ".into(), "\n".into()];
        for (i, s) in specials.iter().enumerate().step_by(97) {
            texts.push(format!("{s}{body}{s}{s} tail {}", specials[(i * 7) % specials.len()]));
            texts.push(format!("{}{s}{}", &body[..300], &body[..200]));
            let half = s.floor_char_boundary(s.len() / 2);
            texts.push(format!("{}{}", &s[..half], &s[half..]));
        }
        let big = format!("<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜></think>{}", body, body.repeat(30), body.repeat(10));
        texts.push(big);
        for round in 0..2 {
            for text in &texts {
                assert_eq!(encode(&tok, text).unwrap(), reference(&tok, text), "round {round}: {text:.60}");
            }
        }
        let c = cache_for(&tok).unwrap().lock().unwrap();
        assert!(!c.entries.is_empty());
    }

    #[test]
    fn eviction_keeps_the_recent_half() {
        let mut c = Cache::default();
        let piece = "y".repeat(CAPACITY_BYTES / 64);
        for i in 0..80 {
            c.clock += 1;
            c.insert(&format!("{i}{piece}"), Arc::from(vec![0u32; 4]));
        }
        assert!(c.bytes <= CAPACITY_BYTES);
        assert!(c.entries.contains_key(&format!("79{piece}")));
        assert!(!c.entries.contains_key(&format!("0{piece}")));
    }
}
