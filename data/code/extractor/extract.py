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
python extract.py

The output will be placed into inter, where folder hp and asoif should exist.
Please note, that the results aren't present in the repository due to their size.
Follow the README.md in the dialogue folder to see how to download the results. 
"""

model_params = {"pipeline": "entity,quote,supersense,event,coref", "model": "big"}

booknlp = BookNLP("en", model_params)

def process_books(path, output_directory, super_id):
    """ 
    Originally this script joined the books into one big file and then processed it.
    However this took over 120GB of RAM and was not feasible. 
    """
    books = os.listdir(path)
    # filter out things not starting with book
    books = [book for book in books if book.startswith("Book")]

    for book in books:
      file_name = path + "/" + book
      book_id = super_id + book.split("-")[1].split('.')[0].replace(" ", "_").lower() 
      print(f"Processing {file_name} to book id {book_id}")
      booknlp.process(file_name, output_directory, book_id)


process_books("./../../books/asoif", "./inter/asoif", "asoif")
process_books("./../../books/hp", "./inter/hp", "hp")
