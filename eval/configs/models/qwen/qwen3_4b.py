from opencompass.models import HuggingFaceCausalLM

# Qwen3 4B setup
# You can download from https://huggingface.co/Qwen/Qwen3-4B
# Make sure to have transformers>=4.40 and accelerate installed.

models = [
    dict(
        abbr="qwen3-4b",
        type=HuggingFaceCausalLM,
        path="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_path="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left"
        ),
        model_kwargs=dict(
            device_map='cuda:0',              # 🚨 QUAN TRỌNG
            torch_dtype="torch.bfloat16",
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=64,                    # ↓ giảm cho GPU đơn
        run_cfg=dict(
            num_gpus=1,
            num_procs=1,
        ),
    ),
]

