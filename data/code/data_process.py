import logging
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
# supresses 'Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.'

print("Imported packages")  # this takes forever on HPC

model_name = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available")
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS is available, Jan is happy")

# model = model.to(device) # Ne dela z bits and bytes


data_path = "./../books/asoif/"
files = os.listdir(data_path)
files = [data_path + file for file in files]

with open(os.path.join(data_path, "characters.txt"), "r", encoding="utf-8") as file:
    characters = ", ".join((map(str.strip, file)))


def extract_dialogues_with_context(filepath, context_length=300):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    pattern = rf"(.{{0,{context_length}}})(“([^”]+)”)(.{{0,{context_length}}})"
    matches = re.finditer(pattern, text, re.DOTALL)

    results = []
    for match in matches:
        pre_context = match.group(1).strip()
        dialogue = match.group(3).strip()  # Group 3 is the actual dialogue within quotes
        post_context = match.group(4).strip()
        results.append(
            {
                "pre_context": pre_context,
                "dialogue": dialogue,
                "post_context": post_context,
            }
        )

    return results


dfs = []
for file in files:
    dfs.extend(extract_dialogues_with_context(file))

df = pd.DataFrame(dfs)

text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)


def classify_dialogues(dialogue_entry):
    # print(dialogue_entry)
    prompt_text = f"""I will give you text before and after a dialogue. Based on this information, you will have to determine who is speaking the dialogue.

### Text before the dialogue: {dialogue_entry['pre_context']} ### End of text before the dialogue.

### Text after the dialogue: {dialogue_entry['post_context']} ### End of text after the dialogue.
Answer with only the cahracter's name. If you are unsure, answer with UNKNOWN.
Who is speaking the dialogue?
### Response: """

    gen = text_gen(prompt_text, max_new_tokens=10)  # [0]["generated_text"].strip()
    print(prompt_text)
    result = gen[0]["generated_text"]
    result = result[len(prompt_text) :].strip()
    result = result.split("\n")[0].replace("### Response: ", "")
    print(result)

    return result


df = df.head(10)

df["character"] = df.apply(classify_dialogues, axis=1)

# print(df.sample(5)[["dialogue", "character"]].values)
