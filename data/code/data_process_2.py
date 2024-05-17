import logging
import os
import re

import nltk
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

nltk.download("punkt")

data_paths = ["./../books/asoif/", "./../books/hp/"]
# data_paths = ["./../books/hp/"]
books: list[tuple[str, str]] = []  # name, path

for data_path in data_paths:
    files = os.listdir(data_path)
    files = [file for file in files if "Book" in file]
    files = [data_path + file for file in files]
    book_names = [file.split("/")[-1].split(".")[0].split("-")[1].strip() for file in files]

    combo = list(zip(book_names, files))

    books.extend(combo)


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+([?.!,:;])", r"\1", text)
    text = re.sub(r"([?.!,:;])\s+", r"\1 ", text)
    return text


def extract_dialogues_with_context(book: tuple[str, str]):
    book_name, filepath = book

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = sent_tokenize(text)
    dialogue_pattern = r"“([^”]+)”"

    c_pre = 10
    c_post = 2

    results = []
    for i, sentence in enumerate(sentences):
        matches = re.findall(dialogue_pattern, sentence)
        for dialogue in matches:
            start_index = max(0, i - c_pre)
            end_index = min(len(sentences), i + c_post + 1)
            context_sentences = sentences[start_index:end_index]
            context = " ".join(context_sentences)
            context = clean_text(context)
            dialogue = clean_text(dialogue)  # Clean dialogue text
            results.append(
                {
                    "Book": book_name,
                    "Context": context,
                    "Dialogue": dialogue,
                }
            )

    return results


dfs = []
for book in books:
    dfs.extend(extract_dialogues_with_context(book))

df = pd.DataFrame(dfs)
df = df.sample(frac=1).reset_index(drop=True)

random_sample = df.sample(3)

for _, row in random_sample.iterrows():
    print(f"--------------- Dialogue from {row['Book']} ---------------")
    print(row["Dialogue"])
    print("--------------- Context ---------------")
    print(row["Context"])
    print()

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


text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)


def classify_dialogues(dialogue_entry):
    prompt_text = f"""
You are a book dialogue character annotator. You will be given a dialogue and its excerpt from the book. Your task is to determine which character from the book is speaking in the dialogue. State only the character's name. If you are unsure, respond with "UNKNOWN".

Examples:

### Who speaks the following dialogue?
"May I come in?"
### In the context of:
She could find Nymeria in the wild woods below the Trident, and together they’d return to Winterfell, or run to Jon on the Wall. She found herself wishing that Jon was here with her now. Then maybe she wouldn’t feel so alone. A soft knock at the door behind her turned Arya away from the window and her dreams of escape. “Arya,” her father’s voice called out. “Open the door. We need to talk.”

Arya crossed the room and lifted the crossbar. Father was alone. He seemed more sad than angry. That made Arya feel even worse. “May I come in?” Arya nodded, then dropped her eyes, ashamed. Father closed the door. “Whose sword is that?”

“Mine.” Arya had almost forgotten Needle, in her hand.

### Answer: Father


### Who speaks the following dialogue?
"Because we were tired and wanted to go to bed,"
### In the context of:
“He knows I’m a Squib!” he finished. “I never touched Mrs. Norris!” Harry said loudly, uncomfortably aware of everyone looking at him, including all the Lockharts on the walls. “And I don’t even know what a Squib is.” “Rubbish!” snarled Filch. “He saw my Kwikspell letter!” “If I might speak, Headmaster,” said Snape from the shadows, and Harry’s sense of foreboding increased; he was sure nothing Snape had to say was going to do him any good. “Potter and his friends may have simply been in the wrong place at the wrong time,” he said, a slight sneer curling his mouth as though he doubted it. “But we do have a set of suspicious circumstances here. Why was he in the upstairs corridor at all? Why wasn’t he at the Halloween feast?” Harry, Ron and Hermione all launched into an explanation about the deathday party. “…there were hundreds of ghosts, they’ll tell you we were there —” “But why not join the feast afterward?” said Snape, his black eyes glittering in the candlelight. “Why go up to that corridor?” Ron and Hermione looked at Harry. “Because — because —” Harry said, his heart thumping very fast; something told him it would sound very far-fetched if he told them he had been led there by a bodiless voice no one but he could hear, “because we were tired and wanted to go to bed,” he said. “Without any supper?” said Snape, a triumphant smile flickering across his gaunt face. “I didn’t think ghosts provided food fit for living people at their parties.” “We weren’t hungry,” said Ron loudly as his stomach gave a huge rumble.

### Answer: Harry

### Who speaks the following dialogue?
"{dialogue_entry['Dialogue']}"
### In the context of:
{dialogue_entry['Context']}
### Answer:
"""

    gen = text_gen(prompt_text, max_new_tokens=10)  # [0]["generated_text"].strip()
    print(prompt_text)
    result = gen[0]["generated_text"]
    result = result[len(prompt_text) :].strip()
    result = result.split("\n")[0].replace("### Response: ", "")
    print(result)

    return result


df = df.head(10)

df["Character"] = df.apply(classify_dialogues, axis=1)

# print(df.sample(5)[["dialogue", "character"]].values)
