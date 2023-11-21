"""Annotate entities, events, and coreference in Litbank books using dbamman/booknlp.
Be sure to use the booknlp environment

Input
-----
litbank-books-directory
    path = litbank/original

Output
------
litbank-annotations-directory
    path = litbank/model-tagged
"""

from booknlp.booknlp import BookNLP
import tqdm
import os

# directories
data_dir = os.getenv("DATA_DIR")
books_dir = os.path.join(data_dir, "litbank/original")
output_dir = os.path.join(data_dir, "litbank/model-tagged")

def annotate_litbank():

    # load booknlp model
    model_params = {"pipeline":"entity,quote,supersense,event,coref", "model":"big"}
    booknlp = BookNLP("en", model_params)

    # load books
    book_text_files = []
    for text_file in os.listdir(books_dir):
        if text_file.endswith(".txt"):
            book_text_files.append(text_file)

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    # tag the books
    for text_file in tqdm.tqdm(book_text_files):
        book_name = text_file[:-4]
        input_file = os.path.join(books_dir, text_file)
        book_output_dir = os.path.join(output_dir, book_name)
        booknlp.process(input_file, book_output_dir, book_name)

if __name__ == '__main__':
    annotate_litbank()