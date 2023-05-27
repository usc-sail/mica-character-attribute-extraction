"""Verify if character descriptions describe character's attribute using alpaca model"""

import os
import math
import tqdm
import torch
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu_id", default=0, help="GPU id")
flags.DEFINE_integer("batch_size", default=32, help="batch size for inference")
flags.DEFINE_integer("sample", default=None, 
                     help="run inference on a sample. If sample is None, run inference on all character descriptions")
flags.DEFINE_string("alpaca_path", default="/project/shrikann_35/llm-shared/alpaca", help="path to alpaca-7B weights")

def alpaca_verify(_):
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

    # load alpaca-7B model
    tokenizer = LlamaTokenizer.from_pretrained(FLAGS.alpaca_path)
    model = LlamaForCausalLM.from_pretrained(FLAGS.alpaca_path, torch_dtype=torch.float16)
    device_id = f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    model.eval()
    model.to(device)

    # prepare dataset
    ids, attrs, prompts = [], [], []
    if FLAGS.sample is not None:
        n = max(0, min(len(character_desc_df), FLAGS.sample))
        character_desc_df = character_desc_df.sample(n)
    character_desc_df["segment_text_size"] = character_desc_df["segment_text"].str.split().apply(len)
    character_desc_df.sort_values(by="segment_text_size", ascending=False, inplace=True)
    for ind, row in tqdm.tqdm(character_desc_df.iterrows(), total=len(character_desc_df), desc="creating alpaca data"):
        character = str(row["character"])
        segment_text = row["segment_text"]
        for attr in attributes:
            ids.append(ind)
            attrs.append(attr)
            prompt = (f"Answer yes/no if the given text describes or mentions {character}'s {attr}.\n\n"
                      f"TEXT: \"{segment_text}\"\nANSWER:")
            prompts.append(prompt)
    print()
    print(f"{len(prompts)} prompts\n")

    # alpaca-7B generate
    print("alpaca-7B generate =>")
    n_batches = math.ceil(len(prompts)/FLAGS.batch_size)
    model.eval()
    answer_texts, answer_token_ids, answer_probs = [], [], []
    with torch.no_grad():
        for i in range(n_batches):
            batch_prompts = prompts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            inputs = tokenizer(batch_prompts, padding="max_length", truncation=True, return_tensors="pt").to("cuda:0")
            input_length = inputs.input_ids.shape[1]
            print(f"\tBatch {i + 1}/{n_batches} input length = {input_length}")
            outputs = model.generate(inputs.input_ids, max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True)
            batch_answers = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            batch_answers = list(map(lambda ans: ans.strip().lower(), batch_answers))
            batch_answer_ids = torch.argmax(outputs.scores[0], dim=1).cpu().tolist()
            batch_scores = torch.softmax(outputs.scores[0], dim=1).max(dim=1).values.cpu().tolist()
            answer_texts.extend(batch_answers)
            answer_token_ids.extend(batch_answer_ids)
            answer_probs.extend(batch_scores)
    print(f"{len(answer_texts)} outputs\n")

    # save outputs: id, attr, answer_text, answer_token_id, answer_prob
    alpaca_df = pd.DataFrame()
    alpaca_df["id"] = ids
    alpaca_df["attr"] = attrs
    alpaca_df["answer_text"] = answer_texts
    alpaca_df["answer_token_id"] = answer_token_ids
    alpaca_df["answer_prob"] = answer_probs
    alpaca_file = os.path.join(data_dir, "alpaca.csv")
    alpaca_df.to_csv(alpaca_file, index=False)

if __name__ == '__main__':
    app.run(alpaca_verify)