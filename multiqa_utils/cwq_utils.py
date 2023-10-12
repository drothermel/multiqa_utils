
## =============================================== ##
## =============== Info Extractors =============== ##
## =============================================== ##

# --------- Original Data Format Extractors ---------- #

def get_original_id(qdata):
    return qdata['ID']

def get_question(qdata):
    return qdata['question']

def get_question_type(qdata):
    multi = len(qdata['answers']) > 1
    if multi:
        return 'complex_multi'
    else:
        return 'complex_single'
    
def get_answer_sets(qdata):
    ann_ans = []
    for ans in qdata['answers']:
        ann_ans.append(list(set([ans['answer'], *ans['aliases']])))
    possible_answer_sets = [ann_ans]
    return possible_answer_sets

# Notes: no gt entities provided
def get_gt_ent_sets(qdata):
    return None

# Notes: None, original retrieval from web broadly
def get_proof_data(qdata, dtk):
    return None
