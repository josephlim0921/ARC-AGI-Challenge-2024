import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import json
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import logging
from unsloth import FastLanguageModel
import torch
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(
    filename="training_log.log",  # Log file name
    level=logging.INFO,           # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)
logger = logging.getLogger(__name__)

# Load model
# max_seq_length = 32768 # Choose any! We auto support RoPE Scaling internally!
max_seq_length = 26000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Hugging face
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# EOS_TOKEN = tokenizer.eos_token

# unsloth
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

logger.info(f"Using model: {MODEL_NAME}")


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the following examples, generate the output array for the provided test input.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompt(examples):
    num_samples = len(examples["input"])
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# def formatting_prompt(examples):
#     inputs = examples["input"]
#     outputs = examples["output"]
#     texts = []
#     for input, output in zip(inputs, outputs):
#         text = f"Given the following examples, generate the output array for the provided test input.\n{input}\n{output}" + EOS_TOKEN
#         texts.append(text)
#     return {"text": texts}


if __name__ == "__main__":

    logger.info("Starting the fine-tuning process.")

    train_df = pd.read_csv("data/train_dataset_cluster6_augmented.csv")
    val_df = pd.read_csv("data/val_dataset.csv")

    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(formatting_prompt, batched=True)
    val_dataset = Dataset.from_pandas(val_df)
    val_dataset = val_dataset.map(formatting_prompt, batched=True)

    logger.info("Datasets prepared. Starting the trainer.")
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     dataset_text_field="text",
    #     dataset_num_proc=2,
    #     packing=True,
    #     args=TrainingArguments(
    #         learning_rate=3e-4,
    #         lr_scheduler_type="linear",
    #         per_device_train_batch_size=2,
    #         gradient_accumulation_steps=8,
    #         num_train_epochs=1,
    #         fp16=not is_bfloat16_supported(),
    #         bf16=is_bfloat16_supported(),
    #         logging_steps=1,
    #         logging_dir="./logs",
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         warmup_steps=10,
    #         output_dir="model",
    #         seed=0,
    #         save_strategy="step",  # Save checkpoints based on epochs
    #         save_total_limit=3,  
    #         dataloader_num_workers=4,
    #         ddp_find_unused_parameters=False,
    #     ),
    # )


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 85,
            #num_train_epochs=8,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "model",
        ),
    )

    trainer_stats = trainer.train()
    metrics = trainer_stats.metrics
    metrics_df = pd.DataFrame([metrics])
    logger.info("Training completed successfully.")

    model_name = MODEL_NAME.split("/")[-1]

    model.save_pretrained(f"model/finetuned_llama_instruct_aug_cluster0_method1_{model_name}")
    tokenizer.save_pretrained(f"model/finetuned_llama_instruct_aug_cluster0_method1_{model_name}")
    metrics_df.to_csv(f"model/finetuned_llama_instruct_aug_cluster0_method1_{model_name}/trainer_stats.csv", index=False)
    logger.info("Model and tokenizer saved successfully.")