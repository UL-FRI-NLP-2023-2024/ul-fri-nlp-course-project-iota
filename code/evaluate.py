import json

from chatbots.book import BookBot
from chatbots.character import CharacterBot
from evaluation.sorting_hat import evaluate_chatbot
from evaluation.trivia import evaluate_trivia
from utils.load_model import load_quantized_pipeline

hat_questions = json.load(open("../data/quizzes/quiz_questions/hp/sorting_hat.json"))
trivial_questions = json.load(open("../data/quizzes/quiz_questions/hp/quiz-questions.json"))
pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")
# pipeline = load_quantized_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")


# bot = BookBot("Harry Potter", use_rag=False, pipeline=pipeline)
# without_rag = evaluate_trivia(bot, trivial_questions)

# bot = BookBot("Harry Potter", use_rag=True, pipeline=pipeline)
# with_rag = evaluate_trivia(bot, trivial_questions)

# print(f"Without RAG: {without_rag}")
# print(f"With RAG: {with_rag}")
for character in ["Hermione"]:
    bot = CharacterBot("Harry Potter", character, use_rag=False, pipeline=pipeline)
    without_rag = evaluate_chatbot(bot, hat_questions)
    bot = CharacterBot("Harry Potter", character, use_rag=True, pipeline=pipeline)
    with_rag = evaluate_chatbot(bot, hat_questions)

    print(f"Character: {character}")
    print(f"Without RAG: {without_rag}")
    print(f"With RAG: {with_rag}")
    print()
