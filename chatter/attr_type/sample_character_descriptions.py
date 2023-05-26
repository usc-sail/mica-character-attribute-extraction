# Sample character descriptions for finding attribute types

import os
import json
import re
import tqdm
import random
import unidecode

from absl import flags
from absl import app

FLAGS = flags.FLAGS
data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
flags.DEFINE_string("scripts_dir", default=os.path.join(data_dir, "scripts"), help="scripts directory")
flags.DEFINE_integer("n", default=500, help="number of character descriptions")
flags.DEFINE_integer("k", default=3, help="number of characters to consider for each movie from the imdb cast list")
flags.DEFINE_string("output_file", default=os.path.join(data_dir, "attr_types/character_descriptions.json"), 
                    help="json file for character descriptions")

def sample_character_descriptions(_):
    character_descriptions = []
    imdb_ids = os.listdir(FLAGS.scripts_dir)

    for imdb_id in tqdm.tqdm(imdb_ids, unit="movie"):
        script_file = os.path.join(FLAGS.scripts_dir, imdb_id, "script.txt")
        parse_file = os.path.join(FLAGS.scripts_dir, imdb_id, "parse.txt")
        imdb_file = os.path.join(FLAGS.scripts_dir, imdb_id, "imdb.json")

        if os.path.exists(script_file) and os.path.exists(parse_file) and os.path.exists(imdb_file):
            with open(script_file, encoding="utf-8") as fr:
                script = fr.read().strip().split("\n")
            with open(parse_file) as fr:
                tags = fr.read().strip().split("\n")
            with open(imdb_file) as fr:
                imdb = json.load(fr)

            descs = []
            i = 0
            while i < len(script):
                if tags[i] == "N":
                    j = i + 1
                    while j < len(script) and tags[j] == tags[i]:
                        j += 1
                    desc = " ".join(script[i: j])
                    desc = re.sub("\s+", " ", desc).strip()
                    descs.append(desc)
                    i = j
                else:
                    i += 1

            characters = []
            if "cast" in imdb:
                for person in imdb["cast"][:FLAGS.k]:
                    if isinstance(person.get("character", None), str) and len(person["character"].split()) > 1 and (
                        re.search("(^|\s)(wo)?man(\s|$)", person["character"].lower()) is None):
                        characters.append(person["character"])
            if not characters:
                continue

            for character in characters:
                pattern = re.compile(f"(^|\s){re.escape(character.lower())}(\s|$)")
                for desc in descs:
                    if re.search(pattern, desc.lower()) is not None:
                        if 50 <= len(desc.split()) <= 200:
                            character_descriptions.append((imdb_id, character, desc))
                        break

    print(f"total {len(character_descriptions)} character descriptions")

    sampled_character_descriptions = random.sample(character_descriptions, k=min(FLAGS.n, len(character_descriptions)))
    sampled_character_descriptions_json = [dict(imdb_id=imdb_id, character=character, desc=unidecode.unidecode(desc)) 
                                           for imdb_id, character, desc in sampled_character_descriptions]
    with open(FLAGS.output_file, "w") as fw:
        json.dump(sampled_character_descriptions_json, fw, indent=2)

    print(f"sampled {len(sampled_character_descriptions)} character descriptions")

if __name__ == '__main__':
    app.run(sample_character_descriptions)