"""Create scene descriptions for characters. 
Scene descriptions are long passages (200 - 496 words) containing the character name.

Input
    - movie scripts directory
        path = mica-movie-scripts/scriptsonscreen/scripts
    - professions csv file
        - path = mica-character-attribute-extraction/professions/soc-mapped-expanded-taxonomy.csv
        - contains list of job titles

Output
    - scene descriptions csv file
        - path = mica-character-attribute-extraction/implicitness/scene_descriptions.csv
        - contains imdb id, scene id, scene description text, and character name
        - scene id is the position of the scene in the script

Parameters
(hard-coded; you will need to edit the code to change them)
    - k
        number of characters per movie
    - min_w
        minimum number of words in a scene description
    - max_w
        maximum number of words in a scene description
"""

import os
import json
import collections
import pandas as pd
import tqdm
import re
import spacy
import unidecode
import torch


k = 5 ## top k characters from IMDB cast list
min_w = 200 ## minimum number of words in scene description
max_w = 496 ## maximum number of words in scene description

# declare paths
data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
scripts_dir = os.path.join(os.getenv("DATA_DIR"), "mica-movie-scripts/scriptsonscreen/scripts")
professions_file = os.path.join(data_dir, "professions/soc-mapped-expanded-taxonomy.csv")
prompt_file = os.path.join(data_dir, "implicitness/scene_descriptions.csv")

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
# data are tuples: (imdb_id, scene)
imdb_id_to_names = collections.defaultdict(list)
data = []

# loop over imdb ids
print("finding scenes =>")
scene_regex = re.compile(r"S+[^S]+")
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
    
    # find scenes
    for match in re.finditer(scene_regex, "".join(tags)):
        start, end = match.span(0)
        segments, segment_tags = [], []

        # find segment and segment tags within scene
        i = start
        while i < end:
            j = i + 1
            while j < end and tags[j] == tags[i]:
                j += 1
            segment = " ".join(script[i: j])
            segment = re.sub("\s+", " ", segment.strip())
            segment = unidecode.unidecode(segment, errors="ignore")
            segments.append(segment)
            segment_tags.append(tags[i])
            i = j

        # create scene segments
        scene_segments = []
        i = 0
        while i < len(segments):
            if segment_tags[i] == "C":
                speaker = segments[i]
                j = i + 1
                utterance_with_expr = []
                while j < len(segments) and segment_tags[j] in ["E", "D"]:
                    segment = segments[j]
                    if segment_tags[j] == "E" and not segment.startswith("("):
                        segment = f"({segment})"
                    utterance_with_expr.append(segment)
                    j += 1
                utterance_with_expr = " ".join(utterance_with_expr)
                utterance_with_expr = re.sub("\s+", " ", utterance_with_expr.strip())
                scene_segment = f"{speaker} says \"{utterance_with_expr}\""
                scene_segments.append(scene_segment)
                i = j
            else:
                scene_segments.append(segments[i])
                i += 1

        scene_text = "\n".join(scene_segments)
        scene_size = len(re.split("\w+", scene_text))
        if min_w <= scene_size <= max_w:
            data.append((imdb_id, scene_text))

    # find imdb cast name list
    if "cast" in imdb_data:
        for person in imdb_data["cast"]:
            if isinstance(person.get("character", None), str):
                name = person["character"]
                if is_valid_name(name, professions):
                    imdb_id_to_names[imdb_id].append(name)

# spacy named entity recognition
print("named entity recognition =>")
scenes = [scene for _, scene in data]
docs = []
for doc in tqdm.tqdm(nlp.pipe(scenes), total=len(scenes), unit="scene"):
    doc._.trf_data = None
    docs.append(doc)
print(f"{len(docs)} spacy docs\n")

# data are tuples: (imdb_id, scene, doc)
data = [(imdb_id, scene, doc) for doc, (imdb_id, scene) in zip(docs, data)]

# prompt_tups are tuples: (imdb_id, scene_id, scene, character)
prompt_tups = []

# loop over imdb ids
for imdb_id in tqdm.tqdm(imdb_ids, unit="movie"):
    scenes_and_docs = [(scene, doc) for x, scene, doc in data if x == imdb_id]
    
    script_names, scene_names = [], []
    for _, doc in scenes_and_docs:
        names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.upper()
                names.append(name)
        script_names.extend(names)
        scene_names.append(set(names))
    
    script_name_to_count = collections.Counter(script_names)
    imdb_names = imdb_id_to_names[imdb_id] if imdb_id in imdb_id_to_names else []

    for i, ((scene, _), names) in enumerate(zip(scenes_and_docs, scene_names)):
        prompt_scene_names = set()
        for name in names:
            name_pattern = re.compile("(^|\W)" + re.escape(name.lower()) + "(\W|$)")
            if (script_name_to_count[name] > 1 
                or any(re.search(name_pattern, imdb_name.lower()) is not None for imdb_name in imdb_names)):
                prompt_scene_names.add(name)
        for name in imdb_names[:k]:
            name_pattern = re.compile("(^|\W)" + re.escape(name.lower()) + "(\W|$)")
            if re.search(name_pattern, scene.lower()) is not None:
                prompt_scene_names.add(name)
        for name in prompt_scene_names:
            prompt_tups.append((imdb_id, i, scene, name))

print(f"{len(prompt_tups)} (imdb_id, scene, character) tuples")
prompt_df = pd.DataFrame(prompt_tups, columns=["imdb_id", "scene_id", "scene_text", "character"])

prompt_df.to_csv(prompt_file, index=False)