"""Create story passages from books.

Input
-----
books-directory
    path = litbank
    litbank/original contains the texts
    litbank/model-tagged contains the model-tagged annotations for entities, events, and coreference

Output
------
books-passages
    path = mica-character-attribute-extraction/book-passages.csv
    The fields are book, passage-id, passage, characters
    passage-id is the paragraph-id in the book

Parameters
----------
min_w
    minimum number of words in passage

max_w
    maximum number of words in passage
"""

import os
import re
import csv
import tqdm
import pandas as pd
from absl import app
from absl import flags

# command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("min_w", default=20, help="minimum number of words")
flags.DEFINE_integer("max_w", default=1000, help="maximum number of words")

# directories and files
data_dir = os.getenv("DATA_DIR")
litbank_dir = os.path.join(data_dir, "litbank")
litbank_texts = os.path.join(litbank_dir, "original")
litbank_annotations = os.path.join(litbank_dir, "tagged") # TODO: change to model-tagged later
passage_file = os.path.join(data_dir, "mica-character-attribute-extraction/book-passages.csv")

def create_passages(_):

    # read books and annotations
    book_names = [text_file[:-4] for text_file in os.listdir(litbank_texts) if text_file.endswith(".txt")]
    books = []
    for book_name in tqdm.tqdm(book_names, unit="book", desc="reading books and annotations"):
        text_file = os.path.join(litbank_texts, book_name + ".txt")
        tokens_file = os.path.join(litbank_annotations, book_name, book_name + ".tokens")
        quotes_file = os.path.join(litbank_annotations, book_name, book_name + ".quotes")
        entities_file = os.path.join(litbank_annotations, book_name, book_name + ".entities")
        with open(text_file, encoding="utf-8") as fr:
            content = fr.read()
        tokens_df = pd.read_csv(tokens_file, index_col=None, sep="\t", quoting=csv.QUOTE_NONE)
        quotes_df = pd.read_csv(quotes_file, index_col=None, sep="\t", quoting=csv.QUOTE_NONE)
        entities_df = pd.read_csv(entities_file, index_col=None, sep="\t")
        books.append({"name": book_name,
                      "text": content,
                      "tokens": tokens_df,
                      "quotes": quotes_df,
                      "entities": entities_df})

    # divide into passages
    passage_tups = []

    # loop over books
    for book in tqdm.tqdm(books, unit="book"):

        # loop over paragraphs
        paragraph_groups = book["tokens"].groupby("paragraph_ID")
        for paragraph_id, paragraph_df in tqdm.tqdm(paragraph_groups, total=paragraph_groups.ngroups, unit="paragraph",
                                                    leave=False):

            # check that paragraph meets word limit
            if FLAGS.min_w < len(paragraph_df) <= FLAGS.max_w:
                token_ids = paragraph_df["token_ID_within_document"].tolist()
                start_token_id, end_token_id = token_ids[0], token_ids[-1]
                byte_onset, byte_offset = paragraph_df.iloc[0]["byte_onset"], paragraph_df.iloc[-1]["byte_offset"]

                # check that paragraph contains no quotes
                internal_quotes_mask = ((start_token_id <= book["quotes"]["quote_start"]) 
                                        & (book["quotes"]["quote_end"] <= end_token_id))

                if internal_quotes_mask.sum() == 0:

                    # check that paragraph contains some person names
                    internal_persons_mask = ((start_token_id  <= book["entities"]["start_token"])
                                             & (book["entities"]["end_token"] <= end_token_id)
                                             & (book["entities"]["cat"] == "PER") 
                                             & (book["entities"]["prop"] == "PROP"))

                    if internal_persons_mask.sum() > 0:
                        passage = book["text"][byte_onset: byte_offset]
                        characters = set()
                        for _, row in book["entities"].loc[internal_persons_mask].iterrows():
                            i = book["tokens"].iloc[row["start_token"]]["byte_onset"]
                            j = book["tokens"].iloc[row["end_token"]]["byte_offset"]
                            character = re.sub("\s+", "", book["text"][i:j])
                            characters.add(character)
                        passage_tups.append([book["name"], paragraph_id, passage, ";".join(sorted(characters))])
    
    # save output
    passage_df = pd.DataFrame(passage_tups, columns=["book", "passage-id", "passage", "characters"])
    passage_df.to_csv(passage_file, index=False)

if __name__ == '__main__':
    app.run(create_passages)