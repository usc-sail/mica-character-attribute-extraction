"""Sample character descriptions for finding attribute types

Input
  - movie scripts folder 
      path = mica-movie-scripts/scriptsonscreen/scripts

Output
  - character descriptions json file
        - path = mica-character-attribute-extraction/attribute-types/character_descriptions.json
        - contains a list and each list entry contains imdb id, character name, and character description
            character name appears in the character description

Parameters
  - n
      number of character descriptions
  - k
      number of characters per movie
"""

import os
import json
import re
import tqdm
import random
import unidecode

from absl import flags
from absl import app

FLAGS = flags.FLAGS
scripts_dir = os.path.join(os.getenv("DATA_DIR"), "mica-movie-scripts/scriptsonscreen/scripts")
output_file = os.path.join(os.getenv("DATA_DIR"), 
                           "mica-character-attribute-extraction/attribute-types/character_descriptions.json")

flags.DEFINE_integer("n", default=500, help="number of character descriptions")
flags.DEFINE_integer("k", default=3, help="number of characters to consider for each movie from the imdb cast list")

def sample_character_descriptions(_):
    character_descriptions = []
    imdb_ids = os.listdir(scripts_dir)

    # loop over movie script directories
    for imdb_id in tqdm.tqdm(imdb_ids, unit="movie"):
        script_file = os.path.join(scripts_dir, imdb_id, "script.txt")
        parse_file = os.path.join(scripts_dir, imdb_id, "parse.txt")
        imdb_file = os.path.join(scripts_dir, imdb_id, "imdb.json")

        # check if movie script directory contains the script file, parse file, and imdb file
        if os.path.exists(script_file) and os.path.exists(parse_file) and os.path.exists(imdb_file):
            with open(script_file, encoding="utf-8") as fr:
                script = fr.read().strip().split("\n")
            with open(parse_file) as fr:
                tags = fr.read().strip().split("\n")
            with open(imdb_file) as fr:
                imdb = json.load(fr)

            # find descriptions (contiguous lines tagges as "N")
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

            # find characters from imdb data
            characters = []
            if "cast" in imdb:
                for person in imdb["cast"][:FLAGS.k]:
                    if isinstance(person.get("character", None), str) and len(person["character"].split()) > 1 and (
                        re.search("(^|\s)(wo)?man(\s|$)", person["character"].lower()) is None):
                        characters.append(person["character"])

            # for each character, find the description first containing the character's name in the movie script
            for character in characters:
                pattern = re.compile(f"(^|\s){re.escape(character.lower())}(\s|$)")
                for desc in descs:
                    if re.search(pattern, desc.lower()) is not None:
                        if 50 <= len(desc.split()) <= 200:
                            character_descriptions.append((imdb_id, character, desc))
                        break

    print(f"total {len(character_descriptions)} character descriptions")

    # sample character descriptions
    sampled_character_descriptions = random.sample(character_descriptions, k=min(FLAGS.n, len(character_descriptions)))
    sampled_character_descriptions_json = [dict(imdb_id=imdb_id, character=character, desc=unidecode.unidecode(desc)) 
                                           for imdb_id, character, desc in sampled_character_descriptions]

    # write to file
    with open(FLAGS.output_file, "w") as fw:
        json.dump(sampled_character_descriptions_json, fw, indent=2)

    print(f"sampled {len(sampled_character_descriptions)} character descriptions")

if __name__ == '__main__':
    app.run(sample_character_descriptions)