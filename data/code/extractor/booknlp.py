import os

from booknlp.booknlp import BookNLP

"""
This is meant to be run on HPC.

ml load Python
ml load CUDA

python3 -m venv venv
source venv/bin/activate
pip install booknlp
python -m spacy download en_core_web_sm
python booknlp.py
"""

model_params = {"pipeline": "entity,quote,supersense,event,coref", "model": "big"}

booknlp = BookNLP("en", model_params)


def process_books(path, output_directory, book_id):
    books = os.listdir(path)
    # filter out things not starting with book
    books = [book for book in books if book.startswith("Book")]

    joined_files = f"{path}/Joined.txt"

    with open(joined_files, "w") as outfile:
        for book in books:
            with open(f"{path}/{book}") as infile:
                outfile.write(infile.read())

    booknlp.process(joined_files, output_directory, book_id)


process_books("./../../books/asoif", "./inter/asoif", "asoif")
process_books("./../../books/hp", "./inter/hp", "hp")
