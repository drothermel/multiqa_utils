
import multiqa_utils.string_utils as su

## =============================================== ##
## =============== Info Extractors =============== ##
## =============================================== ##

# --------- Original Data Format Extractors ---------- #

def get_original_id(qdata):
    return qdata['id']

def get_question(qdata):
    return qdata['question']

def get_question_type(qdata):
    multi = any([
        ann['type'] != 'singleAnswer' for ann in qdata['annotations']
    ])
    if multi:
        return 'simple_multi'
    else:
        return 'simple_single'

# Notes: multiple possible answer sets
def get_answer_sets(qdata):
    possible_answer_sets = []
    for ann in qdata['annotations']:
        if ann['type'] == 'singleAnswer':
            ann_ans = [list(set(ann['answer']))]
            possible_answer_sets.append(ann_ans)
        else:
            ann_ans = []
            for qap in ann['qaPairs']:
                ann_ans.append(list(set(qap['answer'])))
            possible_answer_sets.append(ann_ans)
    return possible_answer_sets

# Notes: viewed_doc_titles is a loose approximation of entities.
def get_gt_ent_sets(qdata):
    ent_sets = set()
    for vd in qdata['viewed_doc_titles']:
        ent_sets.add(frozenset([su.prep_norm(vd), vd]))
    return [list(fs) for fs in ent_sets]

# Notes: None, we just have the results of their queries not the docs used
def get_proof_data(qdata, dtk):
    return None
