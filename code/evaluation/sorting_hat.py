from collections import defaultdict
from typing import TypedDict


class HatAnswer(TypedDict):
    text: str
    values: dict[str, int]


class HatQuestion(TypedDict):
    question: str
    answers: list[HatAnswer]


def evaluate_by_house(pipeline, house: str, hat_questions: list[HatQuestion]):
    scores = defaultdict(int)

    for question in hat_questions:
        prompt = f"""You are a character from the Harry Potter series. You are a person with typical {house} characteristics. Answer the following multiple choice question with only a single letter.\n"""

        prompt += f"\n{question['question']}\n"
        for i, answer in enumerate(question["answers"]):
            prompt += f"({chr(65 + i)}) {answer['text']}\n"

        prompt += "Answer: ("

        output = pipeline(prompt, max_new_tokens=1)[0]["generated_text"]

        model_answer = output[-1]

        for k, v in question["answers"][ord(model_answer) - 65]["values"].items():
            scores[k] += int(v)

    print(f"Desired house: {house} Predicted House: {max(scores, key=scores.get)} ({dict(scores)})")
