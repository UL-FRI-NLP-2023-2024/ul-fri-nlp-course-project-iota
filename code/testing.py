from chatbots.book import BookBot
from utils.load_model import load_quantized_pipeline

pipeline = load_quantized_pipeline("microsoft/phi-2")
bot = BookBot("Harry Potter", use_rag=True, use_summaries=False, pipeline=pipeline)

while True:
    query = input("You: ")
    bot.use_rag = True
    print()
    print(bot.ask(query))
    bot.use_rag = False
    print()
    print(bot.ask(query))
