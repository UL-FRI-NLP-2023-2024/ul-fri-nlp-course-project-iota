from typing import TypedDict

from tqdm import tqdm


class TriviaAnswer(TypedDict):
    text: str
    correct: bool


class TriviaQuestion(TypedDict):
    question: str
    answers: list[TriviaAnswer]


def evaluate_trivia(bot, questions: list[TriviaQuestion]):
    correct = 0

    retires = 5
    for question in tqdm(questions):
        correct_letter = None
        prompt = "Answer the following multiple choice question with only a single letter."

        prompt += f"\n{question['question']}\n"
        answers = ""
        for i, answer in enumerate(question["answers"]):
            prompt += f"({chr(65 + i)}) {answer['text']}\n"
            answers += f"{chr(65 + i)} "
            if answer["correct"]:
                correct_letter = chr(65 + i)

        prompt += f"Answer with a single letter. Available answers: {answers}\n("

        for _ in range(retires):
            output = bot.ask(prompt, max_tokens=1)[-1].upper()

            if output in answers:
                break
        else:
            print("Failed to answer the question.")
            continue

        correct += output == correct_letter

    print(f"Correct: {correct}/{len(questions)} ({correct/len(questions)*100:.2f}%)")
