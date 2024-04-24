import json

from bs4 import BeautifulSoup


def parse_sorting_hat_html(html):
    soup = BeautifulSoup(html)
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


if __name__ == '__main__':
    with open('sorting_hat.html') as f:
        html = f.read()
    questions = parse_sorting_hat_html(html)
    # save to json file
    with open('sorting_hat.json', 'w') as f:
        json.dump(questions, f, indent=2)
