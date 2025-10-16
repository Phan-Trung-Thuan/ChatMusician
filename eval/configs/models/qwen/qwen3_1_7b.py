from opencompass.models import HuggingFaceCausalLM

# Qwen3 1.7B setup
# You can download from https://huggingface.co/Qwen/Qwen3-1.7B
# Make sure to have transformers>=4.40 and accelerate installed.

models = [
    dict(
        abbr="qwen3-1.7b",
        type=HuggingFaceCausalLM,
        path="Qwen/Qwen3-1.7B",        # local folder containing model weights
        tokenizer_path="Qwen/Qwen3-1.7B",  # same path for tokenizer
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        model_kwargs=dict(
            device_map="auto",
            torch_dtype="torch.bfloat16",
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=128,
        run_cfg=dict(num_gpus=2, num_procs=2),
    ),
]
