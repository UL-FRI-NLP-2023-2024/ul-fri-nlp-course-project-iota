from chatbots.book import BookBot
from utils.load_model import load_quantized_pipeline

pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")
bot = BookBot("Harry Potter", pipeline=pipeline)

while True:
    query = input("You: ")
    response = bot.ask(query)
    print(f"Bot: {response}")
