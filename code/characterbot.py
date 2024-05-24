from chatbots.character import CharacterBot
from utils.load_model import load_quantized_pipeline

pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")
bot = CharacterBot("Harry Potter", "Hermione", use_rag=True, pipeline=pipeline)

while True:
    query = input("You: ")
    response = bot.ask(query)
    print(f"Bot: {response}")
