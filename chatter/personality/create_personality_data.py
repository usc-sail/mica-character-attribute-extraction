"""Create personality dataframe by matching title and characters from github.com/YisiSang/Story2Personality to our
imdb scripts data
"""

import os
import re
import json
import tqdm
import pickle
import unidecode
import pandas as pd
from thefuzz import process

def norm(text):
    norm_text = re.sub("\s+", " ", text).lower().strip()
    return unidecode.unidecode(norm_text)

def create_personality():
    # read hero title and characters
    hero_file = os.path.join(os.getenv("DATA_DIR"), 
                             "narrative_understanding/Story2Personality/preprocessed/Movie_superhero.pkl")
    hero_df = pickle.load(open(hero_file, "rb"))
    n_hero_characters = hero_df["id"].unique().size
    print(f"{n_hero_characters} hero characters")
    
    # read script titles and characters
    imdb_ids, imdb_titles, imdb_norm_titles, imdb_characters, imdb_norm_characters = [], [], [], [], []
    n_imdb_characters = 0
    scripts_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter/scripts")
    for imdb_id in os.listdir(scripts_dir):
        imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")
        if os.path.exists(imdb_file):
            imdb_data = json.load(open(imdb_file))
            title = imdb_data["title"]
            characters = []
            if "cast" in imdb_data:
                for person in imdb_data["cast"]:
                    if isinstance(person.get("character", None), str):
                        name = norm(person["character"])
                        characters.append(name)
            if characters:
                imdb_ids.append(imdb_id)
                imdb_titles.append(title)
                imdb_norm_titles.append(norm(title))
                imdb_characters.append(characters)
                imdb_norm_characters.append(list(map(norm, characters)))
                n_imdb_characters += len(characters)
    print(f"{n_imdb_characters} imdb characters")

    # map hero character to imdb character
    # row = imdb_id, imdb_title, imdb_character, pdb_id, pdb_title, pdb_character, fuzzy_title_score, 
    #       fuzzy_character_score, vote_count, I, N, F, P, E, S, T, J
    rows = []
    gb = hero_df.groupby("subcategory")
    for pdb_title, pdb_df in tqdm.tqdm(gb, total=gb.ngroups, unit="pdb movie"):
        pdb_norm_title = norm(pdb_title)
        matched_imdb_norm_title_and_scores = process.extract(pdb_norm_title, imdb_norm_titles, limit=20)
        matched_imdb_norm_title_to_score = dict([(title, score) for title, score in matched_imdb_norm_title_and_scores 
                                                                if score >= 95])
        matched_ids = [i for i, imdb_norm_title in enumerate(imdb_norm_titles) 
                            if imdb_norm_title in matched_imdb_norm_title_to_score.keys()]
        for _, pdb_row in tqdm.tqdm(pdb_df.iterrows(), total=len(pdb_df), unit="pdb character", leave=False):
            pdb_character = pdb_row["mbti_profile"]
            pdb_norm_character = norm(pdb_character)
            for i in matched_ids:
                matched_imdb_norm_character_and_scores = process.extract(pdb_norm_character, imdb_norm_characters[i], 
                                                                         limit=5)
                matched_imdb_norm_character_to_score = dict(
                    [(character, score) for character, score in matched_imdb_norm_character_and_scores
                                            if score >= 95])
                matched_character_ids = [j for j, imdb_norm_character in enumerate(imdb_norm_characters[i])
                                            if imdb_norm_character in matched_imdb_norm_character_to_score.keys()]
                for j in matched_character_ids:
                    rows.append([imdb_ids[i], pdb_row["id"], 
                                 imdb_titles[i], imdb_characters[i][j], 
                                 pdb_title, pdb_character,
                                 matched_imdb_norm_title_to_score[imdb_norm_titles[i]],
                                 matched_imdb_norm_character_to_score[imdb_norm_characters[i][j]]]
                                + pdb_row[["vote_count_mbti", "I", "N", "F", "P", "E", "S", "T", "J"]].tolist())
    
    # save data as dataframe
    personality_df = pd.DataFrame(rows, columns=["imdb_id", "pdb_id", "imdb_title", "imdb_character", 
                                                 "pdb_title", "pdb_character", 
                                                 "fuzzy_title_score", "fuzzy_character_score",
                                                 "vote_count", "I", "N", "F", "P", "E", "S", "T", "J"])
    n_matched_hero_characters = personality_df["pdb_id"].unique().size
    percent = 100 * n_matched_hero_characters/n_hero_characters
    print(f"{n_matched_hero_characters} ({percent:.1f}%) hero characters matched with imdb")
    personality_file = os.path.join(os.getenv("DATA_DIR"), 
                                    "narrative_understanding/chatter/personality/personality.csv")
    personality_df.to_csv(personality_file, index=False)

if __name__=="__main__":
    create_personality()    