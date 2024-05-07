import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

OVERLAP = 64
CHUNK_SIZE = 512

with open("../data/books/hp/hp_preprocessed.txt", "r") as f:
    data = f.readline()

overlaping_chunks = [{"text": data[i : i + CHUNK_SIZE]} for i in range(0, len(data), CHUNK_SIZE - OVERLAP)]
dataset = Dataset.from_list(overlaping_chunks)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
lora_alpha = 32
lora_dropout = 0.1
lora_r = 16
output_dir = "./results"
optim = "paged_adamw_32bit"  # specialization of the AdamW optimizer that enables efficient learning in LoRA setting.
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir=output_dir,
    optim=optim,
    per_device_train_batch_size=2,
    save_total_limit=2,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    load_best_model_at_end=True,
    save_only_model=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
model_to_save.save_pretrained("outputs")
