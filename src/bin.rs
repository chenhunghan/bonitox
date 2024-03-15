use anyhow::{bail, Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::sample::sampler::Sampler;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;

use bonito::parse_a;
use bonito::parse_q;
use bonito::prepare_prompt;
use bonito::TaskType;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The text to generate the question/answer pair from
    #[arg(short = 't', long = "test-chunk")]
    test_chunk: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args { test_chunk } = Args::parse();

    let n_len = 1024;
    let batch_size = 512;

    // only support extractive question answering for now
    let task_type = TaskType::ExtractiveQuestionAnswering;

    let prompt = prepare_prompt(&test_chunk, &task_type);

    let model_repo = "alexandreteles/bonito-v1-gguf";
    // use k-quants because it's faster on metal https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix
    let model_file = "bonito-v1_q4_k_m.gguf";

    // llama.cpp logging flag
    let llama_cpp_log = false;

    let mut llama_cpp_backend = LlamaBackend::init()?;

    if !llama_cpp_log {
        llama_cpp_backend.void_logs();
    }

    let model_params = LlamaModelParams::default();

    let hf_hub_api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(true)
        .build()
        .with_context(|| "unable to create huggingface api")?;

    let hf_model_path = hf_hub_api
        .model(model_repo.to_string())
        .get(&model_file)
        .await?;

    let model = LlamaModel::load_from_file(&llama_cpp_backend, &hf_model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(model.n_ctx_train()));

    // initialize the context
    let mut ctx = model
        .new_context(&llama_cpp_backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // tokenize the prompt
    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_cxt {
        bail!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
        )
    }

    let tokens_list_len = tokens_list.len();

    if tokens_list_len > batch_size {
        bail!(format!("the prompt is too long, it has more tokens than batch_size:{batch_size}"))
    }

    if tokens_list.len() >= usize::try_from(n_len)? {
        bail!(format!("the prompt is too long, it has more tokens than n_len:{n_len}"))
    }

    // create a llama_batch with size `batch_size`
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(batch_size, 1);

    let last_index: i32 = (tokens_list.len() - 1) as i32;

    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last).with_context(|| format!("failed to add token to batch, is your token list length ({tokens_list_len}) bigger than batch size ({batch_size})?"))?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // main loop
    let mut n_cur = batch.n_tokens();

    // completion = prompt + generated string
    let mut completion = String::new();
    completion.push_str(&prompt);

    let finalizer = &|mut canidates: LlamaTokenDataArray, history: &mut Vec<LlamaToken>| {
        canidates.sample_softmax(None);
        let token = canidates.data[0];
        history.push(token.id());
        vec![token]
    };
    let mut history = vec![];
    let mut sampler = Sampler::new(finalizer);

    sampler.push_step(&|c, history| c.sample_repetition_penalty(None, history, 64, 1.1, 0.0, 0.0));
    sampler.push_step(&|c, _| c.sample_top_k(None, 40, 1));
    sampler.push_step(&|c, _| c.sample_tail_free(None, 1.0, 1));
    sampler.push_step(&|c, _| c.sample_typical(None, 1.0, 1));
    sampler.push_step(&|c, _| c.sample_top_p(None, 0.95, 1));
    sampler.push_step(&|c, _| c.sample_min_p(None, 0.05, 1));
    sampler.push_step(&|c, _| c.sample_temp(None, 1.0));

    while n_cur <= n_len {
        // sample the next token
        {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
            let tokens = sampler.sample(&mut history, candidates_p.clone());

            let new_token_id = tokens[0].id();

            if new_token_id == model.token_eos() {
                break;
            }

            let new_str = model.token_to_str(new_token_id)?;
            completion.push_str(&new_str);

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)?;
        }

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;
    }
    ctx.clear_kv_cache();

    let q = parse_q(&completion, &test_chunk);
    if q.is_some() {
        println!("q: {}", q.unwrap());
        println!("a: {}", parse_a(&completion).unwrap());
    } else {
        println!("failed to parse q/a, here is the completion:\n{}", &completion);
    }

    Ok(())
}
