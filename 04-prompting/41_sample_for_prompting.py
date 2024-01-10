"""Sample (segment/scene, attribute, character) tuples for LLM-annotation

Input
    - segment descriptions csv file
        - path = mica-character-attribute-extraction/implicitness/segment_descriptions.csv
        - contains imdb id, segment id, segment description text, and character name
        - segment id is the location of the segment in the movie script
        - created by mica-character-attribute-extraction/implicitness/create_segment_descriptions.py
    - scene descriptions csv file
        - path = mica-character-attribute-extraction/implicitness/scene_descriptions.csv
        - contains imdb id, scene id, scene description text, and character name
        - scene id is the location of the scene in the movie script
        - created by mica-character-attribute-extraction/implicitness/create_scene_descriptions.py
    - flan t5 csv files
        - path = mica-character-attribute-extraction/implicitness/flan_t5/*
        - contains id, attribute-type, answer, answer token id, answer probability (implicitness) fields
        - id here is the dataframe index of the scene/segment description; you can distinguish between scene or segment
            according to attribute-type ("goal" means scene, otherwise segment)
        - created by mica-character-attribute-extraction/implicitness/flan_t5_attr_verify.py
    - demonstrations csv file
        - path = mica-character-attribute-extraction/prompts/samples.csv
        - contains id, passage, character, attribute-type, implicitness
        - id here is the dataframe index, same as the id in flan t5 csv file
        - this file is only used to prevent demonstrations from appearing as samples to be annotated
        - created by mica-character-attribute-extraction/prompt-creation/sample_examples.py

Output
    - samples csv file
        - path = mica-character-attribute-extraction/prompt-results/samples.csv
        - contains attribute-type, id, imdb id, passage id, passage, character name, genres, answer probability
        - id is the dataframe index in the scene/segment descriptions file
        - passage id is the location of the passage (segment/scene) in the movie script

You can change the number of samples in the output samples csv file by varying the number of bins of answer probability
, genre x answer probability bins chosen, and samples per genre x probability bin
Look at lines 90, 151 and 160
"""

import os
import re
import json
import tqdm
import random
import langdetect
import collections
import numpy as np
import pandas as pd

def verify_text(text):
    lines = text.split("\n")
    n_short_lines = sum([len(l.strip().split()) <= 3 for l in lines if l.strip() != ""])
    n_special_characters = np.array([len(re.findall(r"\*", text)), len(re.findall(r"\~", text)), 
                                     len(re.findall(r"\-", text)), len(re.findall(r"\.\.\.", text)),
                                     len(re.findall(r"\:", text))])
    n_digits = len(re.findall("\d", text))
    lang = langdetect.detect(text)
    return (n_short_lines <= 10 
            and sum(n_special_characters) <= 20 
            and np.any(n_special_characters <= 10) 
            and not text.isupper() 
            and n_digits <= 10 
            and lang == "en")

def sample_for_annotation():
    # random seed
    random.seed(0)

    # files and directories
    data_dir = os.path.join(os.getenv("DATA_DIR"), "mica-character-attribute-extraction")
    segment_file = os.path.join(data_dir, "implicitness/segment_descriptions.csv")
    scene_file = os.path.join(data_dir, "implicitness/scene_descriptions.csv")
    flan_files = [os.path.join(data_dir, f"implicitness/flan_t5/flan_t5_{i}_of_16.csv") for i in range(16)]
    demonstrations_file = os.path.join(data_dir, "prompts/samples.csv")
    scripts_dir = os.path.join(os.getenv("DATA_DIR"), "mica-movie-scripts/scriptsonscreen/scripts")
    output_file = os.path.join(data_dir, "prompt-results/samples.csv")

    # read segments, scenes, and few-shot demonstrations
    print("reading segments and scenes ... ", flush=True, end="")
    segment_df = pd.read_csv(segment_file, index_col=None)
    scene_df = pd.read_csv(scene_file, index_col=None)
    demonstrations_df = pd.read_csv(demonstrations_file, index_col=None)
    segment_df["imdb_id"] = segment_df["imdb_id"].astype(str).str.zfill(7)
    scene_df["imdb_id"] = scene_df["imdb_id"].astype(str).str.zfill(7)
    segment_ids_in_demonstrations = demonstrations_df.loc[demonstrations_df["attr"] != "goal", "id"].values
    scene_ids_in_demonstrations = demonstrations_df.loc[demonstrations_df["attr"] == "goal", "id"].values
    print("done")

    # read flan files
    print("reading flan files ... ", flush=True, end="")
    flan_df = pd.concat(pd.read_csv(f, index_col=None) for f in flan_files)
    flan_df["answer_prob_bin"] = np.floor(10*flan_df["answer_prob"])/10
    print("done")

    # read id to imdb id mapping
    print("creating segment/scene dataframe id to imdb id mapping")
    segment_df_id_to_imdb_id = {}
    segment_df_id_to_data = {}
    scene_df_id_to_imdb_id = {}
    scene_df_id_to_data = {}
    for ix, row in tqdm.tqdm(segment_df.iterrows(), total=len(segment_df), unit="segment"):
        segment_df_id_to_imdb_id[ix] = row["imdb_id"]
        segment_df_id_to_data[ix] = (row["segment_text"], row["segment_id"], row["character"])
    for ix, row in tqdm.tqdm(scene_df.iterrows(), total=len(scene_df), unit="scene"):
        scene_df_id_to_imdb_id[ix] = row["imdb_id"]
        scene_df_id_to_data[ix] = (row["scene_text"], row["scene_id"], row["character"])

    # read imdb id to genres mapping
    print("reading genres of movies from imdbpy files ...", flush=True)
    imdb_id_to_genres = {}
    for imdb_id in tqdm.tqdm(os.listdir(scripts_dir), unit="movie"):
        imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")
        try:
            with open(imdb_file) as f:
                imdb_data = json.load(f)
            imdb_id_to_genres[imdb_id] = sorted(set(imdb_data["genres"]))
        except Exception:
            pass

    # sample by genre and difficulty for each attr
    attrs = sorted(flan_df["attr"].unique())
    attr_sampled_ids = []
    attr_sampled_diffs = []
    print("sampling by genre and difficulty for each attribute", flush=True)
    tbar = tqdm.tqdm(attrs, unit="attr")
    for attr in tbar:
        tbar.set_description(attr)
        flan_attr_df = flan_df[(flan_df["attr"] == attr) & (flan_df["answer_text"] == "yes")]
        cat_to_ids = collections.defaultdict(set)
        for _, row in tqdm.tqdm(flan_attr_df.iterrows(), total=len(flan_attr_df), unit="sample", leave=False):
            try:
                if attr == "goal":
                    genres = imdb_id_to_genres[scene_df_id_to_imdb_id[row["id"]]]
                else:
                    genres = imdb_id_to_genres[segment_df_id_to_imdb_id[row["id"]]]
                for genre in genres:
                    cat_to_ids[(genre, row["answer_prob_bin"])].add(row["id"])
            except Exception:
                pass
        for key in cat_to_ids:
            cat_to_ids[key] = sorted(cat_to_ids[key])
        cats = sorted(cat_to_ids.keys())
        cat_counts = []
        for cat in cats:
            count = len(cat_to_ids[cat])
            if count < 10:
                n = 1
            elif 10 <= count < 100:
                n = 2
            else:
                n = 3
            cat_counts.append(n)
        sampled_cats = random.choices(cats, weights=cat_counts, k=350 if attr == "goal" else 300)
        sampled_ids = set()
        for cat in sampled_cats:
            sampled_ids.add(random.choice(cat_to_ids[cat]))
        if attr == "goal":
            sampled_ids.difference_update(scene_ids_in_demonstrations)
        else:
            sampled_ids.difference_update(segment_ids_in_demonstrations)
        sampled_ids = sorted(sampled_ids)
        sampled_ids = sorted(random.sample(sampled_ids, k=min(300 if attr == "goal" else 200, len(sampled_ids))))
        sampled_flan_attr_df = flan_attr_df[flan_attr_df["id"].isin(sampled_ids)]
        flan_attr_id_to_diff = dict((row["id"], row["answer_prob"]) for _, row in sampled_flan_attr_df.iterrows())
        sampled_diffs = [flan_attr_id_to_diff[_id] for _id in sampled_ids]
        attr_sampled_ids.append(sampled_ids)
        attr_sampled_diffs.append(sampled_diffs)
    print()

    # write samples - attr, id, imdb_id, segment/scene id, segment/scene text, character, genres, answer prob
    rows = []
    print("verifying samples for each attribute", flush=True)
    tbar = tqdm.tqdm(zip(attrs, attr_sampled_ids, attr_sampled_diffs), unit="attr", total=len(attrs))
    for attr, sampled_ids, sampled_diffs in tbar:
        tbar.set_description(attr)
        for _id, diff in tqdm.tqdm(zip(sampled_ids, sampled_diffs), unit="sample", leave=False, total=len(sampled_ids)):
            if attr == "goal":
                imdb_id = scene_df_id_to_imdb_id[_id]
                text, text_id, character = scene_df_id_to_data[_id]
            else:
                imdb_id = segment_df_id_to_imdb_id[_id]
                text, text_id, character = segment_df_id_to_data[_id]
            text = re.sub(r"\n{2,}", "\n", text).strip()
            lines = text.split("\n")
            lines = [l for l in lines if re.search(r"[a-zA-Z]", l) is not None]
            text = "\n".join(lines)
            if verify_text(text):
                genres = ",".join(imdb_id_to_genres[imdb_id])
                rows.append((attr, _id, imdb_id, text_id, text, character, genres, diff))
    df = pd.DataFrame(rows, columns=["attr", "id", "imdb_id", "text_id", "text", "character", "genres", "answer_prob"])
    df.to_csv(output_file, index=False)
    print()

    # print genre and difficulty distribution for each attr
    df["answer_prob_bin"] = np.floor(10*df["answer_prob"])/10
    for attr, attr_df in df.groupby("attr"):
        print(f"{attr} = {len(attr_df)} samples")
        genres = np.array([g for gs in attr_df["genres"].str.split(",").tolist() for g in gs])
        genre_dist = [(g, 100 * (genres == g).sum()/len(attr_df)) for g in np.unique(genres)]
        genre_dist = sorted(genre_dist, key=lambda tup: tup[1], reverse=True)
        genre_text = ", ".join(f"{g} ({p:.1f}%)" for g, p in genre_dist)
        print(f"Genres = {genre_text}")
        diff_dist = collections.Counter(attr_df["answer_prob_bin"])
        diff_dist = sorted(diff_dist.items())
        diff_dist = [(diff, 100 * n/(len(attr_df))) for diff, n in diff_dist]
        diff_text = ", ".join(f"{diff} ({p:.1f}%)" for diff, p in diff_dist)
        print(f"Easiness = {diff_text}")
        print()

if __name__=="__main__":
    sample_for_annotation()