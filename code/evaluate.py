import json
from collections import defaultdict

import pandas as pd
from chatbots.book import BookBot
from chatbots.character import CharacterBot
from evaluation.manual_evaluation_questions import (
    asoif_characters_questions,
    asoif_specific_questions,
    harry_potter_characters_questions,
    hp_specific_questions,
    non_specific_questions,
)
from evaluation.sorting_hat import evaluate_hat
from evaluation.trivia import evaluate_trivia
from tqdm import tqdm
from utils.load_model import load_quantized_pipeline


def ask_questions(bot, questions, reveal_character):
    answers = {}
    for question in tqdm(questions):
        answers[question] = bot.ask(question, state_character_and_series=reveal_character, max_tokens=32)

    return answers


def manual_character_eval(series, character, questions, pipeline):
    # Without RAG
    bot = CharacterBot(series, character, pipeline=pipeline, use_rag=False)
    without_rag = ask_questions(bot, questions, reveal_character=True)

    bot = CharacterBot(series, character, pipeline=pipeline, use_rag=True)
    # RAG with revealed character
    with_rag_reveal = ask_questions(bot, questions, reveal_character=True)
    # RAG with hidden character
    with_rag_hide = ask_questions(bot, questions, reveal_character=False)

    output = {}
    for question in questions:
        output[question] = {
            "without_rag": without_rag[question],
            "with_rag_reveal": with_rag_reveal[question],
            "with_rag_hide": with_rag_hide[question],
        }

    with open(
        f"results/character/phi-2/{'hp' if 'Harry' in series else 'asoif'}_{character}_manual_evaluation.json", "w"
    ) as f:
        json.dump(output, f, indent=4)


def create_manual_evaluation_data():
    # pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")
    pipeline = load_quantized_pipeline("microsoft/phi-2")

    character_to_eval = ["Harry", "Dumbledore", "Voldemort"]
    for character in character_to_eval:
        questions = non_specific_questions + hp_specific_questions + harry_potter_characters_questions[character]
        manual_character_eval("Harry Potter", character, questions, pipeline)

    character_to_eval = ["Hodor", "Dany", "Jon"]
    for character in character_to_eval:
        questions = non_specific_questions + asoif_specific_questions + asoif_characters_questions[character]
        manual_character_eval("A Song of Ice and Fire", character, questions, pipeline)


def sorting_hat_eval():
    pipeline = load_quantized_pipeline("microsoft/Phi-3-mini-4k-instruct")

    characters_and_houses = ["Harry", "Dumbledore", "Snape", "Malfoy", "Cho", "Luna", "Cedric", "Tonks"]

    with open("../data/quizzes/quiz_questions/hp/sorting_hat.json") as f:
        hat_questions = json.load(f)

    for character in characters_and_houses:
        bot = CharacterBot("Harry Potter", character, pipeline=pipeline, use_rag=True)

        without_rag_scores = defaultdict(int)
        with_rag_hide_scores = defaultdict(int)
        with_rag_reveal_scores = defaultdict(int)

        without_rag_houses = []
        with_rag_hide_houses = []
        with_rag_reveal_houses = []

        for _ in range(10):
            bot.use_rag = True
            house, scores = evaluate_hat(bot, hat_questions, reveal_character=True)
            with_rag_reveal_houses.append(house)
            for k, v in scores.items():
                with_rag_reveal_scores[k] += v

            house, scores = evaluate_hat(bot, hat_questions, reveal_character=False)
            with_rag_hide_houses.append(house)
            for k, v in scores.items():
                with_rag_hide_scores[k] += v

            bot.use_rag = False
            house, scores = evaluate_hat(bot, hat_questions, reveal_character=True)
            without_rag_houses.append(house)
            for k, v in scores.items():
                without_rag_scores[k] += v

        output = {
            "without_rag": {"houses": without_rag_houses, "scores": dict(without_rag_scores)},
            "with_rag_hide": {"houses": with_rag_hide_houses, "scores": dict(with_rag_hide_scores)},
            "with_rag_reveal": {"houses": with_rag_reveal_houses, "scores": dict(with_rag_reveal_scores)},
        }

        with open(f"results/sorting_hat/{character}_sorting_hat_evaluation.json", "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    # create_manual_evaluation_data()
    sorting_hat_eval()
