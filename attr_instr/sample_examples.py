"""Sample examples for GPT-3.5 prompting to create the demonstrations."""

import os
import re
import numpy as np
import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("n", default=5, help="number of samples per attr per prob bin")

def sample_examples(_):
    data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
    flan_dir = os.path.join(data_dir, "attr_verify/flan_t5_v1")
    flan_files = [os.path.join(flan_dir, f) for f in os.listdir(flan_dir) 
                                            if re.match(r"flan_t5_\d+_of_\d+\.csv", f) is not None]
    segment_file = os.path.join(data_dir, "attr_verify/segment_descriptions.csv")
    scene_file = os.path.join(data_dir, "attr_verify/scene_descriptions.csv")
    output_file = os.path.join(data_dir, "attr_instr/samples.csv")
    flan_df = pd.concat(pd.read_csv(f, index_col=None) for f in flan_files)
    flan_df["answer_prob_bin"] = np.floor(10*flan_df.answer_prob)/10
    segment_df = pd.read_csv(segment_file, index_col=None)
    scene_df = pd.read_csv(scene_file, index_col=None)
    print(f"{len(flan_df)} flan-verified examples")
    samples = []
    for (attr, _), attr_df in flan_df.groupby(["attr", "answer_prob_bin"], sort=True):
        attr_df = attr_df.loc[attr_df.answer_text == "yes"]
        attr_df = attr_df.sample(min(FLAGS.n, len(attr_df)), random_state=0)
        for _, row in attr_df.iterrows():
            text_col, source_df = ("scene_text", scene_df) if attr == "goal" else ("segment_text", segment_df)
            text, character = source_df.loc[row["id"], [text_col, "character"]]
            samples.append([row["id"], text, character, attr, row.answer_prob])
    sample_df = pd.DataFrame(samples, columns=["id", "text", "character", "attr", "answer_prob"])
    print(f"{len(sample_df)} sampled examples")
    sample_df.to_csv(output_file, index=False)

if __name__=="__main__":
    app.run(sample_examples)