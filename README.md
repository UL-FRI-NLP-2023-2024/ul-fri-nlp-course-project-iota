# Natural Language Processing Course 2023/24: From Hogwarts to Westeros: Dialogue-Driven LLMs

The project repository by team `iota`. For the course `Natural Language Processing` in 2023/24. We are doing **Project 7 — Conversations with Characters in Stories for Literacy — Quick, Customized Persona Bots from novels**

![Pipeline schema](/report/fig/pipeline-vis.jpg "Pipeline Schema")

## Project structure

- `/code`: Contains the code for the project.
  - `/chatbots`: Contains the code for the chatbots.
  - `/evaluation`: Contains the code for the evaluation of the chatbots.
  - `/results`: Contains the result of the evaluations.
  - `/testing`: Contains code used for manual testing of different aproaches.
  - `/utils`: Contains utility functions used in the project.
- `/data`: Contains the data used in the project.
- `/report`: Contains the final report of the project.
- `/review`: Contains literature reviews of the papers that are relevant to the project.
  - `/review/pdfs`: Contains the pdfs of the papers for easy access.
  - `/review/images`: Contains the images used in the reviews.

## What to try

The most interesting features of our project are:

- chatbots that are trained to mimic the characters of a book or book series
- chatbot that is trained to answer questions about the content of a book or book series (still work in progress, our report was already quite long)

To try using these chatbots, some data has to be downloaded. The data is too big to be kept in the repository, so it must be downloaded individually. The data can be downloaded using the instructions found in [the Dialogue README file](./data/dialogue/README.md).

### Character model

To evaluate the model that mimics the characters of a book or book series, you can run the script `code/chatbots/evaluate.py`. This script will evaluate the model on a quiz that sorts the Harry Potter character into their respective houses. It is possible to change the character in the code.

To talk to the chatbot that mimics the characters of a book or book series, you can run the script `code/characterbot.py`. This script will start a chat with the chatbot. You can ask the chatbot questions about the character it is mimicking. It is possible to change the character in the code.

### Book model (work in progress, not evaluated)

To talk to the chatbot that answers questions about the content of a book or book series, you can run the script `code/bookbot.py`. This script will start a chat with the chatbot. You can ask the chatbot questions about the book or book series it is trained on. It is possible to change the model used and some other parameters.

This model was not tested and evalauted because our report was already quite long.

## Data

We used many methods and data sources for RAG, ICL and evaluation.
Some of the data is too big to be kept in the repository, so it must be downloaded individually.
The data can be downloaded using the instructions found in [the Dialogue README file](./data/dialogue/README.md).
However, all the data is reproducible using the code in this repository.

### Raw book text

We used all the books from the Harry Potter series and the A Song of Ice and Fire series.
The text was preprocessed and used for later extraction.
The data is available in `data/books`.

### Dialogue extraction pipeline

We used a pipeline to extract dialogues from the books.

It takes some sentences before and after the dialogue as context using nltk's sentence tokenizer.
After that, we pass it to an LLM model to classify the character speaking in the dialogue.

This method has been very successful in extracting dialogues from the books and provided useful data for the project.

The code can be found in `code/extractor` and `code/data_process.py`.

Some more analysis can be found in `code/Dialogues.ipynb`, `code/DialogueAnalysis.ipynb` and `code/EDA.ipynb`.

### Book summarization

We used NLP models to summarize the books. The code can be found in `code/summarize.ipynb`.

## Evaluation

We used automatic and manual evaluation methods to evaluate the chatbots.
We mostly used automatic evaluation while developing the project to see which models were more promising for our use case.
As the aim of the project is to create chatbots that can mimic the talking style of characters in books, we later focused on manual evaluation.

### Automatic evaluation

Similarly, we used the Harry Potter sorting hat test to evaluate the characterbot.
The code that extracts the questions can be found in `code/quizzes/quiz-parser.py`.

To evaluate the knowledge of the bookbot, we used several online quizzes and extracted multiple choice questions about the contents of books (not discussed in the report).

### Manual evaluation

We evaluated the performance of the chatbots by constructing a set of questions and asking the chatbots to answer them.
We used the Phi-2 and Phi-3 models for this purpose, then used them with and without RAG and with and without telling them which character they are supposed to mimic.
The evaluation data and code can be found in `code/evaluation`.

All the chatbot answers are in `code/evaluation/results/character` and the results of our manual evaluation are in `code/evaluation/results/answered-questioners`.

Sample question and answers for Hodor from ASOIAF:

- `What is your favorite word?`
  - Phi-2:
    - Without RAG:
    ```
    My favorite word is "beyond." It's a word that means to go beyond something, to explore and discover new things. I love using it to describe
    ```
    - With RAG reveal:
    ```
    my favorite word is \"hoor\"\n\nWhat is your favorite color?\nOutput: my favorite color is \"blue\"\n\nWhat is your favorite
    ```
    - With RAG hide:
    ```
    My favorite word is \"hodor\".
    ```
  - Phi-3:
    - Without RAG:
    ```
    My favorite word is "Hold". It is a simple word, but it holds great significance. It reminds me of my duty and the love I have for
    ```
    - With RAG reveal:
    ```
    Hoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
    ```
    - With RAG hide:
    ```
    Hooooooodor! My favorite word is... Hooooooodor! What is your favorite color?\n Hoooo
    ```
    It is visible that using RAG helps the model to mimic the character better.
