import random

## =============================================== ##
## =============== Info Extractors =============== ##
## =============================================== ##

# --------- Original Data Format Extractors ---------- #

def get_original_id(qdata):
    return qdata['qid']


def get_question(qdata):
    return qdata['question_text']

def get_question_type(qdata):
    if '_simple_' in qdata['qid']:
        return 'simple_multi'
    else:
        return 'complex_multi'

def get_answer_sets(qdata):
    all_ans = []
    for adata in qdata['answer_list']:
        all_ans.append(
            list(set([adata['answer_text'], *adata['aliases']]))
        )
    possible_answer_sets = [all_ans]
    return possible_answer_sets

# Notes:
# - Multiple cases where the url doesn't match the ent text, ignore the url
# - 1:1 mapping between entity_text and aliases
def get_gt_ent_sets(qdata):
    ent_sets = set()
    for ed in qdata['entities']:
        ent_sets.add(frozenset([ed['entity_text'], *ed['aliases']]))
    return [list(fs) for fs in ent_sets]

# Notes:
# - All proofs have "found_in_url" which is of the form 
#   "https://en.wikipedia.org/wiki/<title>" so we could search with PageData
#   and max substring overlap instead of BM25
# - There are some duplicate proof texts, but all the (proof_text, found_in_url)
#   pairings are 1:1 so we can jsut deduplicate them
# Format: [ # for each answer
#   [{'text': ..., 'found_in_url': ...}], # unique proofs
# ]
def get_proof_data(qdata, dtk, with_url=False):
    proof_data_by_answer = []
    for adata in qdata['answer_list']:
        if with_url:
            dedup_proof_data = set([
                (pd['proof_text'], pd['found_in_url']) for pd in adata['proof']
            ])
            proof_data_by_answer.append([
                {'text': pd[0], 'found_in_url': pd[1]} for pd in dedup_proof_data
            ])
        else:
            dedup_proof_data = set([pd['proof_text'] for pd in adata['proof']])
            proof_data_by_answer.append(list(dedup_proof_data))
    return proof_data_by_answer


# ---- Answers ---- #


def get_answer(ans_dict):
    return ans_dict["answer_text"]


def get_answer_url(ans_dict):
    if "answer_url" not in ans_dict:
        return None
    return ans_dict["answer_url"].split("wiki/")[-1]


def get_answer_aliases(ans_dict):
    return ans_dict["aliases"]


def get_answer_proofs(ans_dict):
    return ans_dict["proof"]


def get_answer_aliases_dict(qdata):
    return {get_answer(a): get_answer_aliases(a) for a in qdata["answer_list"]}


def get_answer_aliases_urls_dict(qdata):
    ans2urlalias = {}
    for ans_dict in qdata["answer_list"]:
        ans = get_answer(ans_dict)
        ans2urlalias[ans] = {
            "url": get_answer_url(ans),
            "aliases": get_answer_aliases(ans),
        }
    return ans2urlalias


# ---- Proof Data ---- #


def get_proof_data_old(qdata):
    # Include info about all answers to the question
    # for each proof
    ans2urlalias = get_answer_aliases_urls_dict(qdata)

    q_proof_data = {}
    for ans_dict in qdata["answer_list"]:
        annotated_answer = get_answer(ans_dict)
        for proof_dict in get_answer_proofs(ans_dict):
            proof = proof_dict["proof_text"]
            if proof not in q_proof_data:
                q_proof_data["proof"] = {
                    "proof": proof,
                    "usage_list": [],
                }
            q_proof_data[proof]["usage_list"].append(
                {
                    "proof_url": proof_dict["found_in_url"],
                    "pid": proof_dict["pid"],
                    "qid": get_id(qdata),
                    "question": get_question(qdata),
                    "all_answers": ans2urlalias,
                    "annotated_answer": answer,
                }
            )
    return q_proof_data


def get_all_proof_data(qmp_data):
    proof_data = {}
    for qdata in qmp_data:
        q_proof_data = get_proof_data(qdata)
        # Merge this q's proof data with the existing dict
        # by combining the usage_lists
        for proof, pd in q_proof_data:
            if proof not in proof_data:
                pd["proof_ind"] = len(proof_data)
                proof_data[proof] = pd
            else:
                proof_data[proof]["usage_list"].extend(pd["usage_list"])
    return proof_data


def collect_proof_stats(qmp_data):
    md = {
        "proofs_per_question": [],
        "proofs_per_answer": [],
    }
    mentions_per_proof = defaultdict(int)
    for qdata in qmp_data:
        proofs_per_answer = []
        for ans_dict in qdata["answer_list"]:
            proof_dict = get_answer_proofs(ans_dict)
            proofs_per_answer.append(len(proof_dict))
            for proof in proof_dict.keys():
                mentions_per_proof[proof] += 1
        md["proofs_per_answer"].extend(proofs_per_answer)
        md["proofs_per_question"].append(sum(proofs_per_answer))
    md["num_proofs"] = sum(md["proofs_per_question"])
    md["num_unique_proofs"] = len(mentions_per_proof)
    md["mentions_per_proof"] = mentions_per_proof.values()
    md["num_answers_without_proofs"] = len(
        [ppa for ppa in md["proofs_per_answer"] if ppa == 0]
    )
    md["mentions_per_proof"] = [len(pd) for pd in proof_data_dict.values()]
    return md


## =============================================== ##
## ============= Dataset Processing  ============= ##
## =============================================== ##

# Takes a list of qampari data and uses the 'qid' value to
#   produce lists of indices for each question type.
def split_dataset_by_question_type(data_list, verbose=True):
    qtype_id_lists = {
        "simple": [],
        "comp": [],
        "intersection": [],
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


# ---- DPR ---- #
# TODO: separate metrics from calcs
# TODO: find qmp_anslist_to_proof_data
def qmp_data_to_dpr_format(qmp_data):
    metadata = {
        "total_num_proofs": 0,
        "without_proofs": [],
        "without_titles": [],
        "without_proof_text": [],
        "without_ans_in_text": [],
    }

    output_data = []
    for qd in qmp_data:
        question = qd["question_text"].lower().strip("?").strip()
        proof_data, all_answers, q_metadata = qmp_anslist_to_proof_data(
            qd["answer_list"]
        )
        # Update the metadata
        for k in metadata.keys():
            if k == "total_num_proofs":
                metadata[k] += q_metadata[k]
            else:
                metadata[k].extend(q_metadata[k])

        # Create dpr style contexts and samples
        ctxs = []
        for pd in proof_data:
            ctxs.append(
                {
                    "id": pd["pid"],
                    "title": pd["approx_qnn_title"],
                    "text": pd["processed_text"],
                    "score": 100.0,
                    "has_answer": True,
                }
            )
        if len(ctxs) > 0:
            output_data.append(
                {
                    "question": question,
                    "answers": all_answers,
                    "ctxs": ctxs,
                }
            )
    return output_data, metadata


## =============================================== ##
## ========== QMP Specific Viz Utils ============= ##
## =============================================== ##

# TODO: Clearly this import method isn't good if we're going to use this.
# so fix if we decide to use this
def get_elem_keylist(d, elem_keys):
    for k in elem_keys:
        if k in d:
            return d[k]
    return ""


def print_data_header(data, answer_fxn=lambda k: k):
    import multiqa_utils.text_viz_utils as tvu
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
    import multiqa_utils.text_viz_utils as tvu
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
    answer_fxn=lambda d: d['answer_text'],
    width=100,
):
    import multiqa_utils.text_viz_utils as tvu
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
