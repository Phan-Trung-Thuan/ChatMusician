from opencompass.models import HuggingFaceCausalLM

# Qwen3 1.7B setup
# You can download from https://huggingface.co/Qwen/Qwen3-1.7B
# Make sure to have transformers>=4.40 and accelerate installed.

models = [
    dict(
        abbr="qwen3-1.7b",
        type=HuggingFaceCausalLM,
        path="./models/qwen/Qwen3-1_7B/",        # local folder containing model weights
        tokenizer_path="./models/qwen/Qwen3-1_7B/",  # same path for tokenizer
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        model_kwargs=dict(
            device_map="auto",
            torch_dtype="auto",
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
