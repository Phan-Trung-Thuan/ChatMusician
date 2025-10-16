from opencompass.models import HuggingFaceCausalLM

# Qwen3 4B setup
# You can download from https://huggingface.co/Qwen/Qwen3-4B
# Make sure to have transformers>=4.40 and accelerate installed.

# Optimized Qwen3 4B config for Kaggle (2x T4 GPUs)
# Make sure you've run:
#   pip install flash-attn --no-build-isolation
#   pip install accelerate transformers>=4.40

models = [
    dict(
        abbr="qwen3-4b",
        type=HuggingFaceCausalLM,
        path="Qwen/Qwen3-4B",               # or local path if downloaded
        tokenizer_path="Qwen/Qwen3-4B",
        tokenizer_kwargs=dict(
            use_fast=True,                  # enables Rust tokenizer
            padding_side='left',
            truncation_side='left'
        ),
        model_kwargs=dict(
            device_map="auto",              # auto-splits across 2 GPUs
            torch_dtype="torch.bfloat16",   # faster and memory-efficient on T4
            # attn_implementation="flash_attention_2",  # use FlashAttention if available
        ),
        max_out_len=128,                    # allow slightly longer generations
        max_seq_len=2048,                   # typical evaluation length
        batch_size=8,                       # safe for T4 memory (4 per GPU)
        run_cfg=dict(
            num_gpus=2,                     # use both T4 GPUs
            num_procs=2,                    # spawn 2 workers
            max_num_workers=2,
        ),
    ),
]