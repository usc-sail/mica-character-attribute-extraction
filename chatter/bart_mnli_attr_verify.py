"""Verify if character descriptions describe character attributes using facebook/bart-large-mnli model"""

import os
import math
import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import BartForSequenceClassification, BartTokenizer

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu_id", default=0, help="GPU id")
flags.DEFINE_integer("batch_size", default=32, help="batch size for inference")
flags.DEFINE_integer("sample", default=None, 
                     help="run inference on a sample. If sample is None, run inference on all character descriptions")
flags.DEFINE_string("hypothesis_template", default="This text describes or mentions CHARACTER's ATTRIBUTE", 
                    help="hypothesis template for nli. CHARACTER and ATTRIBUTE will be replaced by specific values")

def bart_mnli(_):
    # read character descriptions
    data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
    character_desc_file = os.path.join(data_dir, "character_descriptions.csv")
    character_desc_df = pd.read_csv(character_desc_file, index_col=None)
    print(f"{len(character_desc_df)} character descriptions")

    # read attribute types
    attr_type_file = os.path.join(data_dir, "attributes.txt")
    with open(attr_type_file, "r") as fr:
        attributes = fr.read().strip().split("\n")
    attributes = sorted(attributes)
    print(f"{len(attributes)} attribute types")
    print("attribute types =>")
    print(attributes)
    print()

    # load facebook/bart-large-nli model
    print("loading facebook/bart-large-mnli model...", end="", flush=True)
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
    device_id = f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    model.eval()
    model.to(device)
    print("done\n")

    # prepare dataset
    ids, segment_texts, labels = [], [], []
    hypothesis = FLAGS.hypothesis_template
    if FLAGS.sample is not None:
        n = max(0, min(len(character_desc_df), FLAGS.sample))
        character_desc_df = character_desc_df.sample(n)
    character_desc_df["segment_text_size"] = character_desc_df["segment_text"].str.split().apply(len)
    character_desc_df.sort_values(by="segment_text_size", ascending=False, inplace=True)
    for ind, row in tqdm.tqdm(character_desc_df.iterrows(), total=len(character_desc_df), desc="creating nli data"):
        character = str(row["character"])
        for attr in attributes:
            ids.append(ind)
            segment_texts.append(row["segment_text"])
            label = hypothesis.replace("CHARACTER", character).replace("ATTRIBUTE", attr)
            labels.append(label)
    print()
    print(f"{len(segment_texts)} samples\n")

    # nli inference
    print("nli inference =>")
    n_segments = math.ceil(len(segment_texts)/FLAGS.batch_size)
    logits_arr = []
    with torch.no_grad():
        for i in range(n_segments):
            batch_ids = np.array(ids[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]).reshape(-1, 1)
            batch_segment_texts = segment_texts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            batch_labels = labels[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            result = tokenizer(batch_segment_texts, batch_labels, return_tensors="pt", padding="longest")
            input_length = result["input_ids"].shape[1]
            print(f"Batch {i + 1} input length = {input_length}")
            result = result.to(device)
            batch_logits = model(**result)[0]
            batch_logits = np.hstack((batch_ids, batch_logits.cpu().numpy()))
            logits_arr.append(batch_logits)
    print()

    # save logits: num of samples x attributes x 4 [id, entailment, neutral, contradiction]
    logits = np.concatenate(logits_arr).reshape((len(character_desc_df), len(attributes), -1))
    print(f"logits shape = {logits.shape}")
    nli_file = os.path.join(data_dir, "nli.npy")
    np.save(nli_file, logits)

if __name__ == '__main__':
    app.run(bart_mnli)