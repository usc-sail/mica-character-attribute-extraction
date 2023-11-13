"""Create three excel sheets for error analysis of CANNOT ANSWERs, answers other than CANNOT ANSWER, and cot explanations"""

import os
import random
import collections
import pandas as pd

def create_excel():
    # file paths
    data_dir = os.path.join(os.getenv("DATA_DIR"), "narrative_understanding/chatter")
    samples_file = os.path.join(data_dir, "attr_annot/samples.csv")
    zero_file = os.path.join(data_dir, "attr_annot/zero.txt")
    few_file = os.path.join(data_dir, "attr_annot/few.txt")
    cot_answers_file = os.path.join(data_dir, "attr_annot/cot_answers.txt")
    cot_explanations_file = os.path.join(data_dir, "attr_annot/cot_explanations.txt")
    answer_file = os.path.join(data_dir, "attr_analyze/answer.xlsx")
    explanations_file = os.path.join(data_dir, "attr_analyze/explanations.xlsx")

    # read files
    df = pd.read_csv(samples_file, index_col=None)
    strategies = ["zero", "few", "cot"]
    with open(zero_file) as fz, open(few_file) as ff, open(cot_answers_file) as fca, open(cot_explanations_file) as fce:
        df["zero"] = fz.read().strip().split("\n")
        df["few"] = ff.read().strip().split("\n")
        df["cot"] = fca.read().strip().split("\n")
        df["cot_explanation"] = fce.read().strip().split("\n")
    
    # instantiate the excel dataframes
    answer_dfs, answer_sheetnames = [], []
    explanations_dfs, explanation_sheetnames = [], []

    # create answer excel sheets
    n = 0
    for attr, attr_df in df.groupby("attr", sort=True):
        answer_rows = []
        for ix, row in attr_df.iterrows():
            cannotanswer_strategies = []
            answer_to_strategies = collections.defaultdict(list)
            for strategy in strategies:
                if row[strategy].strip().lower() == "cannot answer":
                    cannotanswer_strategies.append(strategy)
                else:
                    answer_to_strategies[row[strategy].strip()].append(strategy)
            if cannotanswer_strategies:
                answer_rows.append([ix, ",".join(cannotanswer_strategies), attr, row["text"], row["character"], 
                                    "CANNOT ANSWER"])
            if answer_to_strategies:
                answer_and_strategies = list(answer_to_strategies.items())
                random.shuffle(answer_and_strategies)
                for answer, ans_strategies in answer_and_strategies:
                    answer_rows.append([ix, ",".join(ans_strategies), attr, row["text"], row["character"], answer])
        answer_df = pd.DataFrame(answer_rows, columns=["index", "strategies", "attribute", "passage", "character", 
                                                       "model_answer"])
        answer_dfs.append(answer_df)
        answer_sheetnames.append(attr)
        print(f"{attr:25s}: {len(answer_df):3d} responses")
        n += len(answer_df)
    print(f"total = {n} answers")
    
    # write excel
    writer = pd.ExcelWriter(answer_file, engine="openpyxl")
    for answer_df, sheetname in zip(answer_dfs, answer_sheetnames):
        answer_df.to_excel(writer, sheet_name=sheetname, index=False)
    writer.close()

if __name__=="__main__":
    create_excel()