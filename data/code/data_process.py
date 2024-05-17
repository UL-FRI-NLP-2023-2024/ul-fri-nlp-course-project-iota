import time

start_time = time.time()

import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import nltk  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from nltk.tokenize import sent_tokenize  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline  # noqa: E402

end_time = time.time()

print(f"Importing packages took {end_time - start_time} seconds")  # For HPC

""" 
This script extracts all the dialoges from HP and ASOIF.
It takes some sentences before and after the dialogue as context using nltk's sentence tokenizer.
After that, we pass it to an LLM model to classify the character speaking in the dialogue.
"""

MIN_DIALOGUE_LENGTH = 10  # minimum length of dialogue to consider
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # model for dialogue classification

nltk.download("punkt")

data_paths = ["./../books/asoif/", "./../books/hp/"]
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

# filter out dialogues that are too short
before_len = len(df)

df = df[df["Dialogue"].apply(len) > MIN_DIALOGUE_LENGTH]

after_len = len(df)

print(f"Filtered out {before_len - after_len} dialogues that were too short. Current length: {after_len}")

random_sample = df.sample(3)

for _, row in random_sample.iterrows():
    print(f"--------------- Dialogue from {row['Book']} ---------------")
    print(row["Dialogue"])
    print("--------------- Context ---------------")
    print(row["Context"])
    print()

logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
# supresses 'Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.'


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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)


def classify_dialogues_batch(dialogue_entries):
    prompts = []
    for dialogue, context in zip(dialogue_entries["Dialogue"], dialogue_entries["Context"]):
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
"{dialogue}"
### In the context of:
{context}

### Answer:
"""
        prompts.append(prompt_text)

    results = text_gen(prompts, max_new_tokens=30)

    characters = []
    for result, dialogue, context in zip(results, dialogue_entries["Dialogue"], dialogue_entries["Context"]):
        result = result[0]
        generated_text = result["generated_text"] if "generated_text" in result else result
        answer = generated_text.split("### Answer:")[-1].strip().split("\n")[0].strip()
        characters.append(answer)

    return {"Character": characters}


df = df.head(20)

print(f"Running on {len(df)} dialogues")

dataset = Dataset.from_pandas(df)
dataset = dataset.map(classify_dialogues_batch, batched=True, batch_size=8)
df = dataset.to_pandas()

df.to_csv("dialogues.csv", index=False)
