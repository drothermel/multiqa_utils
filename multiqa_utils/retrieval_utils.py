import json
import os


# Standardize Filenames

def bm25_out_name(outdir, dataset, split, hits):
    return f"{outdir}/bm25.{dataset}.{split}.h{hits}.json"


# This counts the number of questions in a dataset has no positive included
# (with the assumption that the positive contexts have been produced by some retrieval)
def count_no_positive(dataset_path, hits):
    # In making the datasets I sometimes left the base dataset as only the questions with
    # positive samples in which case the full set is in <filename>__w_no_pos
    no_pos_path = f"{dataset_path}__w_no_pos"
    if os.path.exists(no_pos_path):
        dataset = json.load(open(no_pos_path))
    else:
        dataset = json.load(open(dataset_path))
    total_len = len(dataset)
    no_pos_len = len([d for d in dataset if len(d["positive_ctxs"]) == 0])
    return {"hits": hits, "no_pos_len": no_pos_len, "total_len": total_len}


# And this uses the previous util to display the results
# For example:
# >> ambigqa_bm25_pathname_fxn = lambda hits: ru.bm25_out_name(outdir, "ambigqa_light", "dev", hits)
# >> display_no_positive([100, 400, 1000], ambigqa_bm25_pathname_fxn)
# Hits | No Positive Contexts Retrieved
# ---- | ------------------------------
# 100 | 319/2002 (15.93%)
# 400 | 221/2002 (11.04%)
# 1000 | 170/2002 (8.49%)
#
def display_no_positive(hits_list, pathname_fxn):
    print(f"Hits | No Positive Contexts Retrieved")
    print(f"---- | ------------------------------")
    for hits in hits_list:
        s = count_no_positive(pathname_fxn(hits), hits)
        percent_no_pos = s['no_pos_len'] * 100 / s['total_len']
        print(f"{s['hits']:4} | {s['no_pos_len']}/{s['total_len']} ({percent_no_pos:0.2f}%)")