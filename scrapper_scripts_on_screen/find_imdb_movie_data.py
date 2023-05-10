"""Retrieve imdb movie data using cinemagoer python library and save it as json"""

import os
import imdb
import json
import tqdm
import pandas as pd
from absl import app
from absl import flags

# movie_scripts_dir is the directory where movie scripts (pdfs and texts) are saved, along with their index
movie_scripts_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/movie_scripts/movie_scripts")

# define command-line flags
# --movie_scripts_dir to specify a different movie_scripts_dir
FLAGS = flags.FLAGS
flags.DEFINE_string("movie_scripts_dir", default=movie_scripts_dir, help="Movie Scripts directory.")
flags.DEFINE_bool("overwrite_existing", default=False, help="Overwrite existing data.")

# custon json encoder for imdb movie
class movieEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, imdb.Person.Person):
            if isinstance(o.currentRole, imdb.Character.Character):
                return {"personID": o.personID, "name": o.data.get("name"), 
                        "character": o.currentRole.data.get("name")}
            else:
                return {"personID": o.personID, "name": o.data.get("name")}
        elif isinstance(o, imdb.Company.Company):
            return {"companyID": o.companyID, "name": o.data.get("name")}
        elif isinstance(o, imdb.Movie.Movie):
            return {"movieID": o.movieID, "name": o.data.get("name")}
        return super().default(o)

def find_imdb_movie_data(_):
    imdb_movie_data_file = os.path.join(FLAGS.movie_scripts_dir, "imdb_id_to_movie.json")
    error_imdb_ids_file = os.path.join(FLAGS.movie_scripts_dir, "error_imdb_ids.txt")
    movie_script_index_file = os.path.join(FLAGS.movie_scripts_dir, "index.csv")
    movie_script_index_df = pd.read_csv(movie_script_index_file, index_col=None)
    ia = imdb.Cinemagoer()
    imdb_id_to_movie = {}
    error_imdb_ids = []
    imdb_ids = set(movie_script_index_df["imdb_id"].dropna().str.slice(2).unique().tolist())
    print(f"{len(imdb_ids)} imdb ids in movie script index")

    if not FLAGS.overwrite_existing and os.path.exists(imdb_movie_data_file):
        with open(imdb_movie_data_file, "r") as fr:
            imdb_id_to_movie = json.load(fr)
        n_common_keys = len(imdb_ids.intersection(imdb_id_to_movie.keys()))
        print(f"imdb movie data for {n_common_keys}/{len(imdb_ids)} already retrieved")
        imdb_ids.difference_update(imdb_id_to_movie.keys())

    for imdb_id in tqdm.tqdm(imdb_ids, desc="finding imdb data", unit="movie"):
        try:
            movie = ia.get_movie(imdb_id)
            imdb_id_to_movie[imdb_id] = movie.data
        except Exception:
            error_imdb_ids.append(imdb_id)
    
    json.dump(imdb_id_to_movie, open(imdb_movie_data_file, "w"), indent=2, cls=movieEncoder)
    with open(error_imdb_ids_file, "w") as fw:
        fw.write("\n".join(error_imdb_ids))

if __name__=="__main__":
    app.run(find_imdb_movie_data)