import random
import jsonlines

import multiqa_utils.text_viz_utils as tvu

DOWNLOADED_DATA_DIR = "/scratch/ddr8143/multiqa/downloads/data/qampari/"
PROCESSED_DATA_DIR = "/scratch/ddr8143/multiqa/qampari_data/"
DECOMP_DATA_DIR = f"{PROCESSED_DATA_DIR}decomposition_v0/"
MANUAL_TRAIN_DECOMPOSITION_PATH = f"{DECOMP_DATA_DIR}/manual_decompositions_train.json"


## =============================================== ##
## =============== Info Extractors =============== ##
## =============================================== ##


def extract_answer_text(ans_dict):
    return ans_dict["answer_text"]


def extract_answer_url(ans_dict):
    if "answer_url" not in ans_dict:
        return None
    return ans_dict["answer_url"].split("wiki/")[-1]


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


## =============================================== ##
## ================ Viz Util ================= ##
## =============================================== ##


def get_elem_keylist(d, elem_keys):
    for k in elem_keys:
        if k in d:
            return d[k]
    return ""


def print_data_header(data, answer_fxn=lambda k: k):
    question = get_elem_keylist(data, ["question", "question_text"])
    answers = [
        answer_fxn(a) for a in get_elem_keylist(data, ["answers", "answer_list"])
    ]

    for k, v in {
        "Type": get_elem_keylist(data, ["id", "qid"]),
        "Question": question,
        "Question Keywords": tvu.get_question_keyword_str(question),
        "Answers": tvu.get_answer_str(answers),
    }.items():
        print(f"{ k+':':20} {v}")


def print_retrieval_data(data):
    print_data_header(data)
    for k, v in {
        "Len pos contexts": len(data["positive_ctxs"]),
        "Len ctxs": len(data["ctxs"]),
    }.items():
        print(f"{ k+':':20} {v}")

    tvu.print_ctx_list(
        data["positive_ctxs"],
        answers=data["answers"],
        question=data["question"],
    )


def print_answer_data(
    data,
    answer_fxn=extract_answer_text,
    width=100,
):
    print_data_header(data, answer_fxn)
    answers = data["answer_list"]
    print()
    for ad in answers:
        print(
            "Answer: ", tvu.color_text(ad["answer_text"], "green", [ad["answer_text"]])
        )
        print("    Answer URL:", get_elem_keylist(ad, ["answer_url"]))
        print("    Proofs:")
        for i, proof in enumerate(ad["proof"]):
            p_text = tvu.color_text(
                proof["proof_text"].replace("\n", " "),
                "green",
                [ad["answer_text"].lower()],
            )
            wiki_title = tvu.color_text(
                proof["found_in_url"].split("/wiki/")[-1].replace("_", " "),
                "green",
                [ad["answer_text"]],
            )
            p_text = f"({wiki_title}) {p_text}"
            tvu.print_wrapped(p_text, width)
        print()
