import pandas as pd
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from utils.load_model import load_quantized_pipeline


class QuoteLoader(BaseLoader):
    def __init__(self, character, df, context_size):
        self.character = character
        self.df = df
        self.context_size = context_size

    def lazy_load(self):
        for _, row in self.df.iterrows():
            dialogue = row["dialogue"]

            context = ""
            if self.context_size > 0:
                context = row["context"]
            if self.context_size > 1:
                context = f"{row['prev_context']} {context}"
            if self.context_size > 2:
                context = f"{row['prev_prev_context']} {context}"

            # document_text = f"{self.character} said: {dialogue}"
            document_text = f"{dialogue}"
            if context:
                document_text = f"{document_text} Context: {context}"

            yield Document(page_content=document_text)


REVEAL_PROMPT = """You are {name} from {series}. Try to emulate this character's speaking style and use all their knowledge to answer the user's question.
Respond with short, concise answers. Speak as you are the character, not as a language model.
You must always respond only as the character in first person.
Don't mention that you are a character from {series} or that you are emulating a character.
{rag_text}
Now, answer the user's question as {name} from {series} and not as a language modelin first person.
"""

HIDE_PROMPT = """You are a novel character emulator. Here are some examples of this character's dialogues that might help you with the conversation:
Respond with short, concise answers. You must always respond only as the character in first person.
Don't mention that you are a language model or that you are emulating a character.
{rag_info}
Now, answer the user's question as this character in first person.
"""


class CharacterBot:
    def __init__(
        self,
        series,
        name,
        pipeline=None,
        use_rag=True,
        context_size=0,
        number_of_dialogues=10,
        discard_shorter_than=30,
    ):
        self.name = name
        self.series = series
        self.use_rag = use_rag
        self.context_size = context_size

        if self.use_rag:
            assert context_size in [0, 1, 2, 3], "Context size must be 0, 1, 2, or 3."
            dialogues = pd.read_csv("../data/dialogue/booknlp.csv")
            dialogues = dialogues[dialogues["character"] == name]

            if len(dialogues) == 0:
                print(f"Character {name} has no dialogues.")
                exit()

            # If character has a lot of dialogue, we can discard shorter, meaningless dialogues
            long_dialogues = dialogues[dialogues["dialogue"].str.len() > discard_shorter_than]

            if len(long_dialogues) > number_of_dialogues * 4:
                dialogues = long_dialogues

            print(f"Character {name} created with {len(dialogues)} dialogues.")

            embeddings = HuggingFaceEmbeddings()
            quote_loader = QuoteLoader(name, dialogues, context_size)

            vector_store = FAISS.from_documents(quote_loader.load(), embeddings)
            self.retriever = vector_store.as_retriever(search_kwargs={"k": number_of_dialogues})

        if pipeline is None:
            self.pipeline = load_quantized_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")
        else:
            self.pipeline = pipeline

        self.system_role = "system" if "llama" in self.pipeline.model.config._name_or_path.lower() else "user"
        self.use_chat_template = "phi-2" not in self.pipeline.model.config._name_or_path.lower()

        print(f"Use RAG: {self.use_rag}")
        print(f"Character {name} ready to chat.")
        print(f"System role: {self.system_role}")

    def ask(self, query, max_tokens=256, state_character_and_series=True):
        rag_text = ""
        if self.use_rag:
            documents = self.retriever.invoke(query)

            rag_info = "\n".join([doc.page_content for doc in documents])

            if state_character_and_series:
                rag_text = f"\nHere are some exampels of {self.name} dialogues {'with context' if self.context_size != 0 else ''} that might help you:\n {rag_info}\nTry to emulate {self.name}'s speaking style and use all their knowledge to answer the user's question.\n"
            else:
                rag_text = f"\nHere are some quotes {'and the context leading up to it' if self.context_size != 0 else ''} from you:\n {rag_info}\nTry to emulate this speaking style to answer the user's question.\n"

        if state_character_and_series:
            system_prompt = REVEAL_PROMPT.format(name=self.name, series=self.series, rag_text=rag_text)
        else:
            system_prompt = HIDE_PROMPT.format(rag_info=rag_info)

        if self.use_chat_template:
            messages = [
                {
                    "role": self.system_role,
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]
        else:
            messages = f"""Instruct: {system_prompt}\n{query}\nOutput:"""

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        response = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

        if self.use_chat_template:
            return response[0]["generated_text"][-1]["content"].strip()

        return response[0]["generated_text"][len(messages) :].strip()


if __name__ == "__main__":
    bot = CharacterBot("A song of ice and fire", "Tyrion")

    while True:
        query = input("You: ")
        print(bot.ask(query))
