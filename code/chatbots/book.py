import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from utils.load_model import load_quantized_pipeline


class BookBot:
    def __init__(self, series, retriever_k=2, use_rag=True, use_summaries=True, pipeline=None):
        self.series = series
        self.use_rag = use_rag
        self.use_summaries = use_summaries

        if pipeline is None:
            self.pipeline = load_quantized_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")
        else:
            self.pipeline = pipeline

        if self.use_rag:
            folder = "summaries" if use_summaries else "books"
            series_short = {
                "Harry Potter": "hp",
                "A Song of Ice and Fire": "asoif",
            }[series]
            path = os.path.join("..", "data", folder, series_short)

            text = ""
            for file in os.listdir(path):
                with open(os.path.join(path, file)) as f:
                    text += f.read()

            chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0).split_text(text)

            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_texts(chunks, embeddings)
            self.retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})

        self.system_role = "system" if "llama" in self.pipeline.model.config._name_or_path else "user"

        print(f"BookBot created for {series} series.")
        print(f"RAG: {self.use_rag}")
        print(f"Summaries: {self.use_summaries}")
        print(f"System role: {self.system_role}")

    def ask(self, query, max_tokens=100):
        rag_text = ""

        if self.use_rag:
            documents = self.retriever.invoke(query)

            rag_info = "\n".join([doc.page_content for doc in documents])
            rag_text = f"\nHere are some summaries from the books that might help you:\n{rag_info}\n"

        messages = [
            {
                "role": self.system_role,
                "content": f"""
You are an expert of the {self.series} series.
Answer the user question with all your knowledge about this book series.
Don't repeat the same answer multiple times.
Don't start a conversation with yourself.
End the response when you are done answering the user's question.
Answer in short, concise sentences.
{rag_text}
Now, answer the user's question as an expert of the {self.series} series.""",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

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
    bot = BookBot("Harry Potter")

    while True:
        query = input("You: ")
        response = bot.ask(query)
        print(f"Bot: {response}")
