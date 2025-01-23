import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
import json

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MODEL_NAME = MODEL_NAME.split("/")[-1]
MODEL_PATH = f"model/finetuned_llama_instruct_aug_cluster0_method1_{MODEL_NAME}"

model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = MODEL_PATH,
                        dtype = None,
                        load_in_4bit = True,
                        max_seq_length=32768
                    )
EOS_TOKEN = tokenizer.eos_token

model = FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the following examples, generate the output array for the provided test input.

### Input:
{}

### Response:"""

val_df = pd.read_csv("data/validation_dataset_cluster6_augmented.csv")

print(f"Making predictions with {MODEL_PATH}")
outfile = "eval_finetuned_llama_instruct_aug_cluster0_method1.jsonl"
with open(outfile, "w") as f:
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing Eval Rows"):
        try:
            prompt = alpaca_prompt.format(
                row["input"]
            )
            inputs = tokenizer(
            [
                prompt
            ], return_tensors = "pt").to("cuda")

            max_out_tokens = len(row["output"]) + 40

            outputs = model.generate(**inputs, use_cache=True, max_new_tokens=max_out_tokens, num_return_sequences=2)
            answer = tokenizer.batch_decode(outputs)

            attempt1 = answer[0].split("### Response:")[-1].replace(EOS_TOKEN, "")
            attempt2 = answer[1].split("### Response:")[-1].replace(EOS_TOKEN, "")


            result = {
                "id": row["id"],
                "input_prompt": prompt,
                "output": row["output"],
                "attempt1": attempt1,
                "attempt2": attempt2
            }
            f.write(json.dumps(result) + "\n")
        except Exception as E:
            row_id_print = row["id"]
            print(f"{row_id_print}: {E}")

# outfile = "train_finetuned_v5_llama.jsonl"
# with open(outfile, "w") as f:
#     for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Rows"):
#         try:
#             prompt = alpaca_prompt.format(
#                 row["input"]
#             )
#             inputs = tokenizer(
#             [
#                 prompt
#             ], return_tensors = "pt").to("cuda")

#             max_out_tokens = len(row["output"]) + 40

#             outputs = model.generate(**inputs, use_cache=True, max_new_tokens=max_out_tokens, num_return_sequences=2)
#             answer = tokenizer.batch_decode(outputs)

#             attempt1 = answer[0].split("### Response:")[-1].replace(EOS_TOKEN, "")
#             attempt2 = answer[1].split("### Response:")[-1].replace(EOS_TOKEN, "")


#             result = {
#                 "id": row["id"],
#                 "input_prompt": prompt,
#                 "output": row["output"],
#                 "attempt1": attempt1,
#                 "attempt2": attempt2
#             }
#             f.write(json.dumps(result) + "\n")
#         except Exception as E:
#             print(f"{row["id"]}: {E}")