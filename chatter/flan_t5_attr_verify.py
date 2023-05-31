"""Verify if character descriptions describe character's attribute using google/flan-t5-*"""

import os
import math
import time
import tqdm
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu_id", default=0, help="GPU id")
flags.DEFINE_integer("batch_size", default=32, help="batch size for inference")
flags.DEFINE_integer("sample", default=None, 
                     help="run inference on a sample. If sample is None, run inference on all character descriptions")
flags.DEFINE_integer("start_id", default=None, help="start id of dataset")
flags.DEFINE_integer("end_id", default=None, help="end id of dataset")
flags.DEFINE_enum("t5_model_size", default="small", enum_values=["small", "base", "large", "xl", "xxl"], 
                  help="T5 model size to use")

def timestr(total_seconds: float) -> str:
    time = int(total_seconds)
    seconds = time % 60
    time = time // 60
    minutes = time % 60
    hours = time // 60
    if hours == 0 and minutes == 0:
        return f"{seconds}s"
    elif hours == 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{hours}h{minutes}m{seconds}s"

def flan_t5_verify(_):
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

    # load flan-t5 model
    model_name = f"google/flan-t5-{FLAGS.t5_model_size}"
    print(f"loading {model_name}", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    device_id = f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    model.eval()
    model.to(device)
    print()

    # prepare dataset
    ids, attrs, prompts = [], [], []
    if FLAGS.start_id is not None and FLAGS.end_id is not None:
        end_id = min(FLAGS.end_id, len(character_desc_df))
        character_desc_df = character_desc_df.iloc[FLAGS.start_id: end_id]
        output_file = f"flan_t5_start={FLAGS.start_id}_end={FLAGS.end_id}.csv"
    elif FLAGS.sample is not None:
        n = max(0, min(len(character_desc_df), FLAGS.sample))
        character_desc_df = character_desc_df.sample(n)
        output_file = f"flan_t5_sample={n}.csv"
    else:
        output_file = "flan_t5.csv"
    character_desc_df["segment_text_size"] = character_desc_df["segment_text"].str.split().apply(len)
    character_desc_df.sort_values(by="segment_text_size", ascending=False, inplace=True)
    for ind, row in tqdm.tqdm(character_desc_df.iterrows(), total=len(character_desc_df), desc="creating flan-t5 data"):
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

    # flan-t5 generate
    print("flan-t5 generate =>")
    n_batches = math.ceil(len(prompts)/FLAGS.batch_size)
    model.eval()
    answer_texts, answer_token_ids, answer_probs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for i in range(n_batches):
            batch_prompts = prompts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            inputs = tokenizer(batch_prompts, padding="max_length", truncation=True, return_tensors="pt").to("cuda:0")
            input_length = inputs.input_ids.shape[1]
            print(f"input length = {input_length}, ", end="")
            outputs = model.generate(inputs.input_ids, max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True)
            batch_answers = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            time_elapsed = time.time() - start_time
            time_taken_per_batch = time_elapsed/(i + 1)
            time_to_completion = (n_batches - i - 1) * time_taken_per_batch
            time_taken_per_batch_str = timestr(time_taken_per_batch)
            time_to_completion_str = timestr(time_to_completion)
            print(f"time@batch = {time_taken_per_batch_str}, time left = {time_to_completion_str}")
            batch_answers = list(map(lambda ans: ans.strip().lower(), batch_answers))
            batch_answer_ids = torch.argmax(outputs.scores[0], dim=1).cpu().tolist()
            batch_scores = torch.softmax(outputs.scores[0], dim=1).max(dim=1).values.cpu().tolist()
            answer_texts.extend(batch_answers)
            answer_token_ids.extend(batch_answer_ids)
            answer_probs.extend(batch_scores)
    print(f"{len(answer_texts)} outputs\n")

    # save outputs: id, attr, answer_text, answer_token_id, answer_prob
    flan_t5_df = pd.DataFrame()
    flan_t5_df["id"] = ids
    flan_t5_df["attr"] = attrs
    flan_t5_df["answer_text"] = answer_texts
    flan_t5_df["answer_token_id"] = answer_token_ids
    flan_t5_df["answer_prob"] = answer_probs
    flan_t5_file = os.path.join(data_dir, output_file)
    flan_t5_df.to_csv(flan_t5_file, index=False)

if __name__ == '__main__':
    app.run(flan_t5_verify)