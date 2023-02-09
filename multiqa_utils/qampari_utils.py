import random
import jsonlines

import multiqa_utils.general_utils as gu

DOWNLOADED_DATA_DIR = "/scratch/ddr8143/multiqa/downloads/data/qampari/"
PROCESSED_DATA_DIR = "/scratch/ddr8143/multiqa/qampari_data/"
DECOMP_DATA_DIR = f"{PROCESSED_DATA_DIR}decomposition_v0/"
MANUAL_TRAIN_DECOMPOSITION_PATH = f"{DECOMP_DATA_DIR}/manual_decompositions_train.json"

## =============================================== ##
## ================ Viz Util ================= ##
## =============================================== ##


def print_data(data):
    for k, v in {
        "Type": data["id"],
        "Question": data["question"],
        "Question Keywords": gu.get_question_keyword_str(data["question"]),
        "Answers": gu.get_answer_str(data["answers"]),
        "Len pos contexts": len(data["positive_ctxs"]),
        "Len ctxs": len(data["ctxs"]),
    }.items():
        print(f"{ k+':':20} {v}")

    gu.print_ctx_list(
        data["positive_ctxs"],
        answers=data["answers"],
        question=data["question"],
    )


## =============================================== ##
## ================ General Util ================= ##
## =============================================== ##

# Takes a list of qampari data and uses the 'qid' value to
#   produce lists of indices for each question type.
def split_dataset_by_question_type(data_list, verbose=True):
    qtype_id_lists = {
        "wikidata_simple": [],
        "wikidata_comp": [],
        "wikidata_intersection": [],
    }
    for i, qd in enumerate(data_list):
        for qtype in qtype_id_lists.keys():
            if qtype in qd["qid"]:
                qtype_id_lists[qtype].append(i)

    if verbose:
        for k, v in qtype_id_lists.items():
            print(f"{k + ':':25} {len(v):4}, first 5: {v[:5]}")

    return qtype_id_lists


def random_sample_n_per_type(qtype_ind_list, n, verbose=True):
    random_sample = {}
    for k, indlist in qtype_ind_list.items():
        il = [i for i in indlist]
        random.shuffle(il)
        random_sample[k] = il[:n]

    if verbose:
        for k, v in random_sample.items():
            print(f"{k + ':':25} {len(v):4}, first 5: {v[:5]}")
    return random_sample


def load_dev_data(dpath=f"{DOWNLOADED_DATA_DIR}dev_data.jsonl"):
    qmp_dev = []
    with open(dpath) as f:
        qmp_devd_iter = jsonlines.Reader(f)
        for d in qmp_devd_iter:
            qmp_dev.append(d)
    return qmp_dev


def load_wikidata_dev_data(dpath=f"{DOWNLOADED_DATA_DIR}dev_data.jsonl"):
    qmp_dev = []
    with open(dpath) as f:
        qmp_devd_iter = jsonlines.Reader(f)
        for d in qmp_devd_iter:
            if "wikitables" in d["qid"]:
                continue
            qmp_dev.append(d)
    return qmp_dev


def load_wikidata_train_data(dpath=f"{DOWNLOADED_DATA_DIR}train_data.jsonl"):
    qmp_train = []
    with open(dpath) as f:
        qmp_traind_iter = jsonlines.Reader(f)
        for d in qmp_traind_iter:
            if "wikitables" in d["qid"]:
                continue
            qmp_train.append(d)
    return qmp_train
