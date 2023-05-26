"""Create character descriptions csv"""

import os
import json
import collections
import pandas as pd
import tqdm
import re
import spacy
import unidecode
import torch

from absl import app
from absl import flags


FLAGS = flags.FLAGS
k = 5 ## top k characters from IMDB cast list
min_w = 20 ## minimum number of words in prompt description
max_w = 200 ## maximum number of words in prompt description


def is_valid_name(name: str, professions: set[str]) -> bool:
    """Return true if name refers to a specific named person"""
    general_names = set(["man", "lad", "guy", "adult", "old", "young", "woman", "boy", "girl", "animal", 
                         "child", "lady", "customer", "cafe", "airport", "citizen", "dr."])
    name_parts = set(name.lower().strip().split())
    if len(name_parts.difference(professions).difference(general_names)) == 0:
        return False
    if re.search("[0-9]", name) is not None:
        return False
    if name.startswith("."):
        return False
    return True


# declare paths
data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
scripts_dir = os.path.join(data_dir, "scripts")
professions_file = os.path.join(data_dir, "professions/soc-mapped-expanded-taxonomy.csv")

# collect the imdb ids
imdb_ids = os.listdir(scripts_dir)

# read professions
professions_df = pd.read_csv(professions_file, index_col=None)
professions = set(professions_df["profession"])
print(f"{len(professions)} professions")

# load spacy model, use gpu if available
print("loading spacy...", end="")
if torch.cuda.is_available():
    spacy.require_gpu(0)
nlp = spacy.load("en_core_web_trf", disable=["parser"])
print("done\n")

# imdb_id_to_names is a dictionary of imdb id to list of imdb cast names
# data are tuples: (imdb_id, segment)
imdb_id_to_names = collections.defaultdict(list)
data = []

# loop over imdb ids
print("finding segments =>")
for imdb_id in tqdm.tqdm(imdb_ids, unit="movie"):
    # get file paths to movie script, parse, and imdb data
    script_file = os.path.join(scripts_dir, imdb_id, "script.txt")
    parse_file = os.path.join(scripts_dir, imdb_id, "parse.txt")
    imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")

    # check if the files exist and open
    if os.path.exists(script_file) and os.path.exists(parse_file) and os.path.exists(imdb_file):
        with open(script_file, encoding="utf-8") as fr:
            script = fr.read().strip().split("\n")
        with open(parse_file) as fr:
            tags = fr.read().strip().split("\n")
        with open(imdb_file, encoding="utf-8") as fr:
            imdb_data = json.load(fr)
        
        # find segments from movie script
        i = 0
        while i < len(script):
            if tags[i] == "N":
                j = i + 1
                while j < len(script) and tags[j] == tags[i]:
                    j += 1
                segment = " ".join(script[i: j])
                segment = re.sub("\s+", " ", segment).strip()
                segment = unidecode.unidecode(segment, errors="ignore")
                if min_w <= len(segment.split()) <= max_w:
                    data.append((imdb_id, segment))
                i = j
            else:
                i += 1

        # find imdb cast name list
        if "cast" in imdb_data:
            for person in imdb_data["cast"]:
                if isinstance(person.get("character", None), str):
                   name = person["character"]
                   if is_valid_name(name, professions):
                       imdb_id_to_names[imdb_id].append(name)

print(f"{len(data)} (imdb_id, segment) pairs\n")

# spacy named entity recognition
print("named entity recognition =>")
segments = [segment for _, segment in data]
docs = []
for doc in tqdm.tqdm(nlp.pipe(segments), total=len(segments), unit="segment"):
    doc._.trf_data = None
    docs.append(doc)
print(f"{len(docs)} spacy docs\n")

# data are tuples: (imdb_id, segment, doc)
data = [(imdb_id, segment, doc) for doc, (imdb_id, segment) in zip(docs, data)]

# prompt_tups are tuples: (imdb_id, segment_id, segment, character)
prompt_tups = []

# loop over imdb ids
for imdb_id in tqdm.tqdm(imdb_ids, unit="movie"):
    segments_and_docs = [(segment, doc) for x, segment, doc in data if x == imdb_id]
    
    script_names, segment_names = [], []
    for _, doc in segments_and_docs:
        names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.upper()
                names.append(name)
        script_names.extend(names)
        segment_names.append(set(names))
    
    script_name_to_count = collections.Counter(script_names)
    imdb_names = imdb_id_to_names[imdb_id] if imdb_id in imdb_id_to_names else []

    for i, ((segment, _), names) in enumerate(zip(segments_and_docs, segment_names)):
        prompt_segment_names = set()
        for name in names:
            name_pattern = re.compile("(^|\W)" + re.escape(name.lower()) + "(\W|$)")
            if (script_name_to_count[name] > 1 
                or any(re.search(name_pattern, imdb_name.lower()) is not None for imdb_name in imdb_names)):
                prompt_segment_names.add(name)
        for name in imdb_names[:k]:
            name_pattern = re.compile("(^|\W)" + re.escape(name.lower()) + "(\W|$)")
            if re.search(name_pattern, segment.lower()) is not None:
                prompt_segment_names.add(name)
        for name in prompt_segment_names:
            prompt_tups.append((imdb_id, i, segment, name))

print(f"{len(prompt_tups)} (imdb_id, segment, character) tuples")
prompt_df = pd.DataFrame(prompt_tups, columns=["imdb_id", "segment_id", "segment_text", "character"])

prompt_file = os.path.join(data_dir, "character_descriptions.csv")
prompt_df.to_csv(prompt_file, index=False)