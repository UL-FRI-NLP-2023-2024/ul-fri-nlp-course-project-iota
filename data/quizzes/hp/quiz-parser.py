import json

from bs4 import BeautifulSoup


def parse_sorting_hat_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    cards = soup.findAll('div', {'class': 'card-body'})
    questions = []
    for card in cards:
        question = card.find('h5').text
        answers = []
        for answer in card.findAll('div', {'class': 'form-check'}):
            if answer.find('label').text.strip() == 'Blank':
                continue
            answers.append({
                'text': answer.find('label').text.strip(),
                'values': {
                    'gryffindor': answer.find('input')['data-g'],
                    'hufflepuff': answer.find('input')['data-h'],
                    'ravenclaw': answer.find('input')['data-r'],
                    'slytherin': answer.find('input')['data-s'],
                },
            })
        questions.append({
            'question': question,
            'answers': answers,
        })
    return questions

def parse_ultimate_hp_quiz_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    questions = []
    for question in soup.findAll('div', {'class': 'question'}):
        question_text = question.find('div', {'class': 'text-2xl mb-4'}).text.strip()
        answers = []
        for answer in question.findAll('div', {'class': 'flex justify-between items-center'}):
            answer_val = answer.find('div').text.strip()
            if 'incorrect-icon' in answer.find('svg').attrs['class']:
                correct = False
            else:
                correct = True
            answers.append({
                'text': answer_val,
                'correct': correct,
            })
        questions.append({
            'question': question_text,
            'answers': answers,
        })
    return questions

        

if __name__ == '__main__':
    # with open('sorting_hat.html') as f:
    #     html = f.read()
    # questions = parse_sorting_hat_html(html)
    # save to json file
    # with open('sorting_hat.json', 'w') as f:
    #     json.dump(questions, f, indent=2)
    with open('smeti/ultimate-hp-quiz.html') as f:
        html = f.read()
    questions = parse_ultimate_hp_quiz_html(html)
    # save to json file
    with open('smeti/ultimate_hp_quiz.json', 'w') as f:
        json.dump(questions, f, indent=2)
