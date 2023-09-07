import re
import string
import regex
from functools import lru_cache
from operator import itemgetter

from sacremoses import MosesDetokenizer

import utils.file_utils as fu

DATASETS = [
    ("qmp", "qampari"),
    ("rqa", "romqa"),
    ("qst", "quest"),
]

# ---------- General Utils ----------- #

def flatten_list(in_list):
    return [item for row in in_list for item in row]

# From qampari, models/evaluation/retriever_metric.py
def longest_common_substring(x: str, y: str) -> (int, int, int):
    # function to find the longest common substring

    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1)
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:

        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0

    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():

        # upper right triangle of the 2D array
        for k in range(len(x)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(y) - 1, -1, -1)))

        # lower left triangle of the 2D array
        for k in range(len(y)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(x) - 1, -1, -1)))

    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))


# ---------- Normalization Utils ----------- #

def get_detokenizer():
    return MosesDetokenizer(lang="en")

# Normalization used by qmp when loading in wiki title and text
# So the chunks already have this and we should apply it to the
# candidate strs before returning them as answers.
def normalize(detokenizer, el):
    el = fix_qu(el.replace("'", "'"))
    tokens = el.split(" ")
    return detokenizer.detokenize(tokens).replace("'", "'")


def fix_qu(string):
    pat = re.compile('"(.*?)"')
    pat2 = re.compile('" (.*?) "')
    pat3 = re.compile("'(.*?)'")
    pat4 = re.compile("' (.*?) '")
    for x in pat.finditer(string):
        to_replace = x.group(0)
        res = pat2.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace, replace_with)
    for x in pat3.finditer(string):
        to_replace = x.group(0)
        res = pat4.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace, replace_with)
    return string

# The version of norm that qampari provides in
#     models/evaluation/reader_metrics.py
def qmp_norm(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None or s == "":
        return s

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# My current best guess at the full normalization applied by qampari
def qnn_norm(s):
    return qmp_norm(normalize(s))


# ---------- Dataset Access Utils ----------- #

def data_to_dict(data_list, data_name):
    return {get_data_field(data_name, "id", d): d for d in data_list}


def get_data_field(data_type, field, data_elem):
    def qmp_answers(d):
        return [(a["answer_text"], a["aliases"]) for a in d["answer_list"]]

    def rqa_answers(d):
        return [(a["text"], a["aliases"]) for a in d["complete_answer"]]

    def qst_answers(d):
        return [(a, [a]) for a in d["docs"]]

    key_tuples = {
        "question": [
            ("qmp", "question_text"),
            ("qampari", "question_text"),
            ("rqa", "question"),
            ("romqa", "question"),
            ("qst", "query"),
            ("quest", "query"),
        ],
        "id": [
            ("qmp", "qid"),
            ("qampari", "qid"),
            ("rqa", "id"),
            ("romqa", "id"),
            ("qst", "id"),
            ("quest", "id"),
        ],
        "answers": [
            ("qmp", qmp_answers),
            ("qampari", qmp_answers),
            ("rqa", rqa_answers),
            ("romqa", rqa_answers),
            ("qst", qst_answers),
            ("quest", qst_answers),
        ],
    }

    if field in ["question", "id"]:
        for data_t, data_k in key_tuples[field]:
            if data_t in data_type:
                return data_elem[data_k]

    if field in ["answer", "answers"]:
        for data_t, data_k_lambda in key_tuples["answers"]:
            if data_t in data_type:
                return data_k_lambda(data_elem)
    print(f"data_type: {data_type}")
    assert False


# path_cfg: must contain elq_datasets and base_datasets
def get_data(path_cfg, data_type, as_dict=False, verbose=False):
    if "simple" in data_type or "complex" in data_type:
        assert False, "Unimpld for now"

    assert "train" in data_type or "dev" in data_type
    assert not ("train" in data_type and "dev" in data_type)
    split = "train" if "train" in data_type else "dev"

    base_data = []
    for dshort, dfull in DATASETS:
        if dshort in data_type or dfull in data_type:
            base_data.extend(
                fu.load_file(path_cfg[dfull][split], verbose=verbose)
            )
    if len(base_data) is []:
        raise Exception(f"ERROR: Invalid data type: {data_type}")

    if as_dict:
        if "elq" in data_type:
            return {d["id"]: d for d in base_data}
        else:
            return data_to_dict(base_data, data_type)
    return base_data


# ---------- Data Access Utils ----------- #

# ---- QMP Proof info ---- #
def qmp_anslist_to_proof_data(
    ans_list,
    filter_by_answer_presence=True,
    filter_by_title_presence=True,
):
    metadata = {
        'total_num_proofs': 0,
        'without_proofs': [],
        'without_titles': [],
        'without_proof_text': [],
        'without_ans_in_text': [],
    }
    proof_data = []
    found_answers = []
    for ad in ans_list:
        if len(ad['proof']) == 0:
            metadata['without_proofs'].append(ad)
            continue
        qnn_answer = qnn_norm(ad['answer_text'])
        num_found = 0
        for pd in ad['proof']:
            if 'found_in_url' not in pd:
                metadata['without_titles'].append((ad, pd))
                title = ''
                if filter_by_title_presence:
                    continue
            else:
                title = pd['found_in_url'].split('/')[-1].replace('_', ' ')

            if 'proof_text' not in pd:
                metadata['without_proof_text'].append((ad, pd))
                continue
            metadata['total_num_proofs'] += 1

            text = pd['proof_text']
            if qnn_answer not in qnn_norm(title + ' ' + text):
                metadata['without_ans_in_text'].append((ad, pd, text))
                if filter_by_answer_presence:
                    continue

            processed_text = text
            if text[0] == ' ':
                processed_text = title + text
            processed_text = qnn_norm(processed_text)
            num_found += 1

            proof_data.append({
                'qid': '_'.join(pd['pid'].split('_')[:-1]),
                'pid': pd['pid'],
                'approx_qnn_title': qnn_norm(title),
                'text': text,
                'processed_text': processed_text,
                'qnn_answer': qnn_answer,
            })
        if num_found > 0:
            found_qnn_answers.append(qnn_answer)
    return proof_data, found_qnn_answers, metadata


# Output: { proof: [proof_data_for_questionN_answerM,, ...], ...}
def qmp_raw_to_proof_info(qmp_data):
    md = {
        'proofs_per_question': [],
        'num_answers_without_proofs': 0,
    }

    proof_data_dict = {}
    proof_inds = {}
    for qd in qmp_data:
        num_q_proofs = 0
        # Get question data to store
        qid = qd['qid']
        question = qd['question_text']
        all_answers = {
            ad['answer_text']: {
                'aliases': ad['aliases'],
                'url': ad['answer_url'] if 'answer_url' in ad else None
            } for ad in qd['answer_list']
        }

        # Add all proof data for this q
        for ad in qd['answer_list']:
            answer = ad['answer_text']
            if len(ad['proof']) == 0:
                md['num_answers_without_proofs'] += 1
            for pd in ad['proof']:
                num_q_proofs += 1
                proof = pd['proof_text']
                proof_url = pd['found_in_url']
                if proof not in proof_data_dict:
                    proof_data_dict[proof] = []
                    proof_inds[proof] = len(proof_inds)
                proof_data_dict[proof].append({
                    'proof_ind': proof_inds[proof],
                    'proof_url': pd['found_in_url'],
                    'pid': pd['pid'],
                    'qid': qid,
                    'question': question,
                    'all_answers': all_answers,
                    'annotated_answer': answer,
                })
        md['proofs_per_question'].append(num_q_proofs)
    md['num_proofs'] = sum(md['proofs_per_question'])
    md['num_unique_proofs'] = len(proof_data_dict)
    md['mentions_per_proof'] = [len(pd) for pd in proof_data_dict.values()]
    return proof_data_dict, md


# Deterministic
def qmp_proof_data_to_query_list(qmp_proof_data):
    proof_query_list = [
        {
            'proof': proof,
            'pid': str(pds[0]['proof_ind']),
            'usage_list': pds,
        } for proof, pds in qmp_proof_data.items() if len(pds) > 0
    ]
    proof_query_list = sorted(
        proof_query_list, key=lambda d: d['pid'],
    )
    return proof_query_list

# ---- QMP to DPR Format ---- #
def qmp_data_to_dpr_format(qmp_data):
    metadata = {
        'total_num_proofs': 0,
        'without_proofs': [],
        'without_titles': [],
        'without_proof_text': [],
        'without_ans_in_text': [],
    }

    output_data = []
    for qd in qmp_data:
        question = qd['question_text'].lower().strip('?').strip()
        proof_data, all_answers, q_metadata = qmp_anslist_to_proof_data(
            qd['answer_list']
        )
        # Update the metadata
        for k in metadata.keys():
            if k == 'total_num_proofs':
                metadata[k] += q_metadata[k]
            else:
                metadata[k].extend(q_metadata[k])

        # Create dpr style contexts and samples
        ctxs = []
        for pd in proof_data:
            ctxs.append({
                'id': pd['pid'],
                'title': pd['approx_qnn_title'],
                'text': pd['processed_text'],
                'score': 100.0,
                'has_answer': True,
            })
        if len(ctxs) > 0:
            output_data.append({
                'question': question,
                'answers': all_answers,
                'ctxs': ctxs,
            })
    return output_data, metadata

def dpr_out_to_eval_format(data):
    eval_data = {
        'gold_ans': [],
        'preds': [],
        'scores': [],
        'labels': [],
    }
    for qdata in data:
        q_gold_ans = set(qdata['gold_answers'])
        eval_data['gold_ans'].append(q_gold_ans)
        eval_data['preds'].append([p['prediction_text'] for p in qdata['predictions']])
        eval_data['scores'].append([p['span_score'] for p in qdata['predictions']])
        eval_data['labels'].append([
            int(p['prediction_text'] in q_gold_ans) for p in qdata['predictions']
        ])
    return eval_data


# ---- Entity Linking Metrics ---- #

# Parse elq data consistently (no accidental flips!)
def get_elq_entoriqnn(edata):
    return [
        {
            "ent": ee[0],
            "ori": ee[1],
            "qnn_ent": wu.qnn_norm(ee[0]),
            "qnn_ori": wu.qnn_norm(ee[1]),
        }
        for ee in edata["pred_tuples_string"]
    ]


# Return dict for this question: {ent: (url, aliases_set)}
def get_qmp_entities(elem, good_only=False):
    ent2urlalias = {}
    for ent_dict in elem["entities"]:
        ent = ent_dict["entity_text"]
        if ent not in ent2urlalias:
            ent_url = ent_dict["entity_url"] if "entity_url" in ent_dict else None
            if good_only and ent_url is None:
                continue
            ent2urlalias[ent] = (ent_url, set())
        ent2urlalias[ent][1].update([a for a in ent_dict["aliases"]])
    return ent2urlalias


def get_rqa_entities(elem, good_only=False):
    # Note that rqa entities are always good due to dataset construction
    ent2urlalias = {}
    for c in elem["constraints"]:
        ent = c["other_ent"]["text"]
        if ent not in ent2urlalias:
            ent2urlalias[ent] = (c["other_ent"]["uri"], set())
        ent2urlalias[ent][1].update(c["other_ent"]["aliases"])
    return ent2urlalias


# Returns: {id: {gt_ent: [gt_ent_aliases]}}
def get_gt_ent2aliases(dname, data_dict):
    if dname == "quest" or dname == "qst":
        return None
    assert dname in ["rqa", "romqa", "qmp", "qampari"]

    all_ent2aliases = {}
    for qid, qdata in data_dict.items():
        if dname == "rqa" or dname == "romqa":
            ent2urlalias = get_rqa_entities(qdata)
        else:  # dname == 'qmp' or dname == 'qampari'
            ent2urlalias = get_qmp_entities(qdata)
        all_ent2aliases[qid] = {k: v[1] for k, v in ent2urlalias.items()}
    return all_ent2aliases



if __name__  == '__main__':
    print("Loading")
    qmp_dev_path = '/scratch/ddr8143/multiqa/downloads/data/qampari/dev_data.jsonl'
    qmp_dev = fu.load_file(qmp_dev_path)

    print("Running")
    test_qd = [qd for i, qd in enumerate(qmp_dev) if i < 10]
    proof_data = qmp_raw_to_proof_info(test_qd)
    proof_query_list = qmp_proof_data_to_query_list(proof_data)
    #breakpoint()
