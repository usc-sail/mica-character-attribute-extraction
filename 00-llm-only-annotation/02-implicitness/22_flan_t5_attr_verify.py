"""Verify if segment/scene descriptions describe character's attribute using google/flan-t5-*
This gives us the implicitness score (refer to paper)

Input
    - segment descriptions csv file
        - path = mica-character-attribute-extraction/implicitness/segment_descriptions.csv
        - contains imdb id, segment id, segment description text, and character name
        - segment id is location of segment in movie script
        - produced by create_segment_descriptions.py
    - scene descriptions csv file
        - path = mica-character-attribute-extraction/implicitness/scene_descriptions.csv
        - contains imdb id, scene id, scene description text, and character name
        - scene id is location of scene in movie script
        - produced by create_scene_descriptions.py
    - prompt template txt file
        - path = mica-character-attribute-extraction/implicitness/flan_t5_prompt_templates.txt
        - contains the prompt to verify if segment/scene description contains attribute-value

Output
    - flan answers csv file
        - path = mica-character-attribute-extraction/implicitness/flan_t5/*
            suffix will be flan_t5_sample=n.csv if sample is specified, otherwise suffix will be flan_t5_x_of_y.csv
            n is the number of samples, x is split id, and y is total number of splits (refer to parameters)
        - contains id, attribute-type, answer, answer token id, answer probability fields
        - id is the index in segment/scene descriptions file; if attribute-type is "goal" then it is the index in the
            scene descriptions file, otherwise segment descriptions file

Parameters
    - gpu id
    - inference batch size
    - number of descriptions to sample
    - total number of splits
    - split id
    - t5 model size
"""

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
                     help="run inference on a sample. If sample is None, run inference on all descriptions")
flags.DEFINE_integer("n_data_splits", default=16, help="number of data splits")
flags.DEFINE_integer("split_id", default=0, help="data split on which to run inference")
flags.DEFINE_enum("t5_model_size", default="small", enum_values=["small", "base", "large", "xl", "xxl"], 
                  help="T5 model size to use")
flags.register_multi_flags_validator(["n_data_splits", "split_id"], 
                                     lambda args: 0 <= args["split_id"] < args["n_data_splits"],
                                     "split_id < n_data_splits")

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
    """Prompt flan-t5 model to check if description describes/specifies character attribute"""
    # read segment and scene descriptions
    data_dir = os.path.join(os.getenv("DATA_DIR"), "mica-character-attribute-extraction/implicitness")
    segment_desc_file = os.path.join(data_dir, "segment_descriptions.csv")
    scene_desc_file = os.path.join(data_dir, "scene_descriptions.csv")
    segment_desc_df = pd.read_csv(segment_desc_file, index_col=None)
    scene_desc_df = pd.read_csv(scene_desc_file, index_col=None)
    print(f"{len(segment_desc_df)} segment descriptions")
    print(f"{len(scene_desc_df)} scene descriptions")

    # read attribute types and their prompt templates
    attr_template_file = os.path.join(data_dir, "flan_t5_prompt_templates.txt")
    attrs_and_templates = []
    with open(attr_template_file, "r") as fr:
        lines = fr.read().strip().split("\n")
        i = 0
        while i < len(lines) - 1:
            attr = lines[i].strip()
            template = lines[i + 1].strip()
            attrs_and_templates.append((attr, template))
            i += 3
    attrs_and_templates = sorted(attrs_and_templates)
    attrs = [attr for attr, _ in attrs_and_templates]
    print(f"{len(attrs_and_templates)} attribute types")
    print("attribute types => ")
    print(attrs)
    print()

    # load flan-t5 model
    model_name = f"google/flan-t5-{FLAGS.t5_model_size}"
    print(f"loading {model_name}", flush=True)
    start_time = time.time()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    device_id = f"cuda:{FLAGS.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    model.eval()
    model.to(device)
    time_taken = timestr(time.time() - start_time)
    print(f"time taken to load = {time_taken}\n")

    # prepare dataset - find subset index
    if FLAGS.sample is not None:
        n_scene = math.ceil(FLAGS.sample / len(attrs))
        n_segment = FLAGS.sample - n_scene
        n_scene = max(0, min(len(scene_desc_df), n_scene))
        n_segment = max(0, min(len(segment_desc_df), n_segment))
        scene_desc_df = scene_desc_df.sample(n_scene)
        segment_desc_df = segment_desc_df.sample(n_segment)
        print(f"sample data size = {len(segment_desc_df)} segments + {len(scene_desc_df)} scenes")
        output_file = f"flan_t5_sample={n_scene + n_segment}.csv"
    else:
        segment_split_size = math.ceil(len(segment_desc_df)/FLAGS.n_data_splits)
        segment_desc_df = segment_desc_df.iloc[FLAGS.split_id * segment_split_size: 
                                               (FLAGS.split_id + 1) * segment_split_size]
        scene_split_size = math.ceil(len(scene_desc_df)/FLAGS.n_data_splits)
        scene_desc_df = scene_desc_df.iloc[FLAGS.split_id * scene_split_size: (FLAGS.split_id + 1) * scene_split_size]
        print(f"data size = {len(segment_desc_df)} segments + {len(scene_desc_df)} scenes")
        output_file = f"flan_t5_{FLAGS.split_id}_of_{FLAGS.n_data_splits}.csv"
    
    # prepare dataset - create segment prompts for flan-t5 model
    # data is list of (segment/scene id, attr, prompt, prompt size)
    data = []
    for ind, row in tqdm.tqdm(segment_desc_df.iterrows(), total=len(segment_desc_df), 
                              desc="creating flan-t5 segment data"):
        character = str(row["character"])
        segment_text = row["segment_text"]
        for attr, template in attrs_and_templates:
            if attr != "goal":
                instruction = template.replace("<CHARACTER>", character)
                prompt = f"{instruction}\n\nTEXT: \"{segment_text}\"\nANSWER:"
                data.append((ind, attr, prompt, len(prompt.split())))
    
    # prepare dataset - create scene prompts for flan-t5 model
    for ind, row in tqdm.tqdm(scene_desc_df.iterrows(), total=len(scene_desc_df), 
                              desc="creating flan-t5 scene data"):
        character = str(row["character"])
        segment_text = row["scene_text"]
        for attr, template in attrs_and_templates:
            if attr == "goal":
                instruction = template.replace("<CHARACTER>", character)
                prompt = f"{instruction}\n\nTEXT: \"{segment_text}\"\nANSWER:"
                data.append((ind, attr, prompt, len(prompt.split())))
    
    # prepare dataset - sort by prompt size
    data = sorted(data, key=lambda item: item[3], reverse=True)
    print(f"{len(data)} prompts\n")

    # flan-t5 generate - outputs is a list of (answer text, answer token id, answer prob)
    print("flan-t5 generate =>")
    outputs = []

    # flan-t5 generate - loop over batches
    model.eval()
    start_time = time.time()
    n_batches = math.ceil(len(data)/FLAGS.batch_size)
    with torch.no_grad():
        for i in range(n_batches):
            batch_data = data[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            batch_prompts = [item[2] for item in batch_data]
            inputs = tokenizer(batch_prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            print(f"batch {i + 1}/{n_batches}: input length = {input_length}, ", end="")
            batch_outputs = model.generate(inputs.input_ids, max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True)
            batch_answer_texts = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
            
            # estimate time to completion
            time_elapsed = time.time() - start_time
            time_taken_per_batch = time_elapsed/(i + 1)
            time_to_completion = (n_batches - i - 1) * time_taken_per_batch
            time_taken_per_batch_str = timestr(time_taken_per_batch)
            time_to_completion_str = timestr(time_to_completion)
            print(f"time@batch = {time_taken_per_batch_str}, time left = {time_to_completion_str}", flush=True)

            # find text, token id, and prob of answer
            batch_answer_texts = list(map(lambda ans: ans.strip().lower(), batch_answer_texts))
            batch_answer_token_ids = torch.argmax(batch_outputs.scores[0], dim=1).cpu().tolist()
            batch_answer_probs = torch.softmax(batch_outputs.scores[0], dim=1).max(dim=1).values.cpu().tolist()
            batch_outputs = []
            for answer_text, answer_token_id, answer_prob in zip(batch_answer_texts, batch_answer_token_ids, 
                                                                 batch_answer_probs):
                batch_outputs.append((answer_text, answer_token_id, answer_prob))
            outputs.extend(batch_outputs)
    print(f"{len(outputs)} outputs\n")

    # save outputs: id, attr, answer_text, answer_token_id, answer_prob
    flan_t5_data = []
    for (ind, attr, _, _), (answer_text, answer_token_id, answer_prob) in zip(data, outputs):
        flan_t5_data.append((ind, attr, answer_text, answer_token_id, answer_prob))
    flan_t5_df = pd.DataFrame(flan_t5_data, columns=["id", "attr", "answer_text", "answer_token_id", "answer_prob"])
    flan_t5_file = os.path.join(data_dir, "flan_t5", output_file)
    flan_t5_df.to_csv(flan_t5_file, index=False)

if __name__ == '__main__':
    app.run(flan_t5_verify)