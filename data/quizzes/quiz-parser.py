import json

from bs4 import BeautifulSoup


def parse_sorting_hat_html(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.findAll("div", {"class": "card-body"})
    questions = []
    for card in cards:
        question = card.find("h5").text
        answers = []
        for answer in card.findAll("div", {"class": "form-check"}):
            if answer.find("label").text.strip() == "Blank":
                continue
            answers.append(
                {
                    "text": answer.find("label").text.strip(),
                    "values": {
                        "gryffindor": answer.find("input")["data-g"],
                        "hufflepuff": answer.find("input")["data-h"],
                        "ravenclaw": answer.find("input")["data-r"],
                        "slytherin": answer.find("input")["data-s"],
                    },
                }
            )
        questions.append(
            {
                "question": question,
                "answers": answers,
            }
        )
    return questions


def parse_ultimate_hp_quiz_html(html):
    soup = BeautifulSoup(html, "html.parser")
    questions = []
    for question in soup.findAll("div", {"class": "question"}):
        question_text = question.find("div", {"class": "text-2xl mb-4"}).text.strip()
        answers = []
        for answer in question.findAll("div", {"class": "flex justify-between items-center"}):
            answer_val = answer.find("div").text.strip()
            if "incorrect-icon" in answer.find("svg").attrs["class"]:
                correct = False
            else:
                correct = True
            answers.append(
                {
                    "text": answer_val,
                    "correct": correct,
                }
            )
        questions.append(
            {
                "question": question_text,
                "answers": answers,
            }
        )
    return questions


def parse_quiz_questions_json(questions):
    new_questions = []
    for question in questions:
        question_text = question["question"]
        answers = []
        for answer in question["possibleAnswers"]:
            answers.append(
                {
                    "text": answer["answer"],
                    "correct": answer["correctAnswer"],
                }
            )
        new_questions.append(
            {
                "question": question_text,
                "answers": answers,
            }
        )
    return new_questions


def parse_hp_quiz(html):
    soup = BeautifulSoup(html, "html.parser")
    questions = []
    for question in soup.findAll("div", {"class": "grid-item"}):
        if question.find("div", {"class": "qcard-question"}) is None:
            continue
        question_text = question.find("div", {"class": "qcard-question"}).text.strip()
        answers = []
        for answer in question.findAll("div", {"class": ["qcard-answer-multi", "qcard-answer-multi-correct"]}):
            answers.append(
                {
                    "text": answer.text.strip(),
                    "correct": "qcard-answer-multi-correct" in answer["class"],
                }
            )
        questions.append(
            {
                "question": question_text,
                "answers": answers,
            }
        )
    return questions


def parse_aqoiaf(html):
    soup = BeautifulSoup(html, "html.parser")
    questions = []
    for question in soup.findAll("li", {"class": "ques_marg"}):
        text = question.find("h3").text.strip()
        answers = []
        answer_items = question.findAll("li")
        correct_answer = question.find("div", {"class": "correct_ans_list"}).text.strip().split(".")[0][-1]
        for answer in answer_items:
            mark = answer.find("div", {"class": "questonnopt"}).text.strip().split(".")[0][-1]
            answers.append(
                {
                    "text": answer.find("p").text.strip(),
                    "correct": correct_answer == mark,
                }
            )

        questions.append(
            {
                "question": text,
                "answers": answers,
            }
        )
    return questions


if __name__ == "__main__":
    with open("src/hp/sorting_hat.html") as f:
        html = f.read()
    questions = parse_sorting_hat_html(html)
    # save to json file
    with open("quiz_questions/hp/sorting_hat.json", "w") as f:
        json.dump(questions, f, indent=2)

    with open("src/hp/ultimate-hp-quiz.html") as f:
        html = f.read()
    questions = parse_ultimate_hp_quiz_html(html)
    print(len(questions))

    with open("src/hp/quiz-questions.json") as f:
        data = json.load(f)
    questions += parse_quiz_questions_json(data)
    print(len(questions))

    for i in range(1, 4):
        with open(f"src/hp/hp-quiz-pg{i}.htm") as f:
            html = f.read()
        questions += parse_hp_quiz(html)

    print(len(questions))

    with open("quiz_questions/hp/quiz-questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    asoiaf_questions = []
    with open("src/asoiaf/aqoiaf-4.htm") as f:
        html = f.read()
    asoiaf_questions.append(parse_aqoiaf(html))
    with open("src/asoiaf/aqoiaf-5.htm") as f:
        html = f.read()
    asoiaf_questions.append(parse_aqoiaf(html))
    with open("quiz_questions/asoiaf/quiz_questions.json", "w") as f:
        json.dump(asoiaf_questions, f, indent=2)
