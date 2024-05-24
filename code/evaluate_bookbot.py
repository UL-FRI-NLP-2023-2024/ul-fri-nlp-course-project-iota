import json

from chatbots.book import BookBot
from evaluation.trivia import evaluate_trivia
from utils.load_model import load_quantized_pipeline

hat_questions = json.load(open("../data/quizzes/quiz_questions/hp/sorting_hat.json"))
trivia_questions = json.load(open("../data/quizzes/quiz_questions/hp/quiz-questions.json"))
pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")
# pipeline = load_quantized_pipeline("meta-llama/Meta-Llama-3-8B-Instruct")


bot = BookBot("Harry Potter", use_rag=False, pipeline=pipeline)
without_rag = evaluate_trivia(bot, trivia_questions)

bot = BookBot("Harry Potter", use_rag=True, pipeline=pipeline)
with_rag = evaluate_trivia(bot, trivia_questions)
