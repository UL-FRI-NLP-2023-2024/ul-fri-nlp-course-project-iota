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
            document_text = f"Chracter said: {dialogue}"
            if context:
                document_text = f"{document_text} In context of: {context}"

            yield Document(page_content=document_text)


class CharacterBot:
    def __init__(
        self, series, name, pipeline=None, use_rag=True, context_size=0, number_of_dialogues=30, discard_shorter_than=30
    ):
        self.name = name
        self.series = series
        self.use_rag = use_rag

        if self.use_rag:
            assert context_size in [0, 1, 2, 3], "Context size must be 0, 1, 2, or 3."
            dialogues = pd.read_csv("../data/dialogue/booknlp.csv")
            dialogues = dialogues[dialogues["character"] == name]

            if len(dialogues) == 0:
                print(f"Character {name} has no dialogues.")
                exit()

            dialogues = dialogues[dialogues["dialogue"].str.len() > discard_shorter_than]

            print(f"Character {name} created with {len(dialogues)} dialogues.")

            embeddings = HuggingFaceEmbeddings()
            quote_loader = QuoteLoader(name, dialogues, context_size)

            vector_store = FAISS.from_documents(quote_loader.load(), embeddings)
            self.retriever = vector_store.as_retriever(search_kwargs={"k": number_of_dialogues})

        if pipeline is None:
            self.pipeline = load_quantized_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")
        else:
            self.pipeline = pipeline

        self.system_role = "system" if "llama" in self.pipeline.model.config._name_or_path else "user"

        print(f"Use RAG: {self.use_rag}")
        print(f"Character {name} ready to chat.")
        print(f"System role: {self.system_role}")

    def ask(self, query, max_tokens=256):
        rag_text = ""
        if self.use_rag:
            documents = self.retriever.invoke(query)

            rag_info = "\n".join([doc.page_content for doc in documents])
            # print(rag_info)
            rag_text = f"\nHere are some quotes and the context leading up to it from {self.name} that might help you:\n {rag_info}\n"
            # rag_text = f"\nHere are some exampels of {self.name} dialogues that might help you:\n {rag_info}\ Try to emulate {self.name}'s speaking style and use all their knowledge to answer the user's question.\n"

        messages = [
            {
                "role": self.system_role,
                "content": f"""
        You are {self.name} from {self.series}.
        Use only the following examples to answer the user's question.
        {rag_text}
        Now, answer the user's question as {self.name} from {self.series}.
        """,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        #         messages = [
        #             {
        #                 "role": self.system_role,
        #                 "content": f"""
        # You are a novel character emulator. Here are some examples of this character's dialogues that might help you:
        # {rag_info}
        # Now, answer the user's question as this character.
        # """,
        #             },
        #             {
        #                 "role": "user",
        #                 "content": query,
        #             },
        #         ]

        #         print(messages)

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        response = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            top_p=0.95,
        )

        return response[0]["generated_text"][-1]["content"].strip()


if __name__ == "__main__":
    bot = CharacterBot("A song of ice and fire", "Tyrion")

    while True:
        query = input("You: ")
        print(bot.ask(query))
