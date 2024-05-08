import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

print("Imported packages") # this takes forever on HPC

# Load the Phi-3 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)

if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model moved to GPU")
else:
    print("CUDA is not available, using CPU")


data_path = "./../books/asoif/"
files = os.listdir(data_path)
files = [data_path + file for file in files]

def extract_dialogues_with_context(filepath, context_length=100):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    pattern = rf"(.{{0,{context_length}}})(“([^”]+)”)(.{{0,{context_length}}})"
    matches = re.finditer(pattern, text, re.DOTALL)

    results = []
    for match in matches:
        pre_context = match.group(1).strip()
        dialogue = match.group(3).strip()  # Group 3 is the actual dialogue within quotes
        post_context = match.group(4).strip()
        results.append({
            "pre_context": pre_context,
            "dialogue": dialogue,
            "post_context": post_context,
        })

    return results

dfs = []
for file in files:
    dfs.extend(extract_dialogues_with_context(file))

df = pd.DataFrame(dfs)
print(df.shape)

pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


def classify_dialogues(dialogue_entry):
    prompt_text = f"Given the context and dialogue, determine which character is speaking. Output only the character name. If you are not sure, return UKNOWN. Context: '{dialogue_entry['pre_context']}' Dialogue: '{dialogue_entry['dialogue']}' Following Context: '{dialogue_entry['post_context']}'"
    
    result = pipeline(prompt_text, max_new_tokens=30)

    return result[0]['generated_text'].strip()

df = df.head(20)

df['character'] = df.apply(classify_dialogues, axis=1)

print(df.sample(5)[["dialogue", "character"]].values)