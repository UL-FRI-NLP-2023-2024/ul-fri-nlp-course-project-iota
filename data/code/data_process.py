import logging
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
# supresses 'Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.'

print("Imported packages")  # this takes forever on HPC

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available")
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS is available, Jan is happy")

model = model.to(device)


data_path = "./../books/asoif/"
files = os.listdir(data_path)
files = [data_path + file for file in files]


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
    prompt_text = f"I need you to identify who is speaking in the provided dialoge. I will give you context surrounding the dialogue, to help you figure it out. Context: '{dialogue_entry['pre_context']}' Dialogue: '{dialogue_entry['dialogue']}' Following Context: '{dialogue_entry['post_context']}' Who is speaking?"

    gen = text_gen(prompt_text, max_new_tokens=30)  # [0]["generated_text"].strip()

    print(gen)

    result = gen[0]["generated_text"].strip()

    print(result[-50:])

    return result


df = df.head(1)

df["character"] = df.apply(classify_dialogues, axis=1)

# print(df.sample(5)[["dialogue", "character"]].values)
