from typing import TypedDict

from tqdm import tqdm


class TriviaAnswer(TypedDict):
    text: str
    correct: bool


class TriviaQuestion(TypedDict):
    question: str
    answers: list[TriviaAnswer]


def evaluate_trivia(pipeline, questions: list[TriviaQuestion]):
    correct = 0

    for question in tqdm(questions):
        correct_letter = None
        prompt = "Answer the following multiple choice question with only a single letter."

        prompt += f"\n{question['question']}\n"
        for i, answer in enumerate(question["answers"]):
            prompt += f"({chr(65 + i)}) {answer['text']}\n"
            if answer["correct"]:
                correct_letter = chr(65 + i)
        prompt += "Answer: ("

        output = pipeline(prompt, max_new_tokens=1)[0]["generated_text"]

        model_answer = output[-1]

        correct += model_answer == correct_letter

        if not model_answer == correct_letter:
            print(f"{output}")
            print(f"Correct answer: {correct_letter}")
            print()

    print(f"Correct: {correct}/{len(questions)} ({correct/len(questions)*100:.2f}%)")
