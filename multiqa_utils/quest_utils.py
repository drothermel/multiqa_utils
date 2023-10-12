
import multiqa_utils.string_utils as su

## =============================================== ##
## =============== Info Extractors =============== ##
## =============================================== ##

# --------- Original Data Format Extractors ---------- #

def get_original_id(qdata):
    return None

def get_question(qdata):
    return qdata['query']

def get_question_type(qdata):
    return 'complex_multi'

# QST
# Notes:
# - no aliases, there are evidence_ratings and relevance_ratings for each
#   doc that provides confidence that this is an answer and the attribution
#   provides all the info you would need to get the answer.
# - Make the "prefix answer" an alias because they are doing prediction based
#   on classifying document names but we're doing generation/entity identification
#   so it seems only fair.
def get_answer_sets(qdata):
    all_ans = [list(set([d, su.prep_norm(d)])) for d in qdata['docs']]
    possible_answer_sets = [all_ans]
    return possible_answer_sets

def get_gt_ent_sets(qdata):
    return None

# Notes:
# - Run normalize on the docs if we want any chance of getting them.
def get_proof_data(qdata, dtk):
    proof_data_by_answer = []
    for doc in qdata['docs']:
        if qdata['metadata']['attributions'] == None:
            proof_data_by_answer.append([])
            continue

        doc_data_list = qdata['metadata']['attributions'][doc]
        all_proofs = set()
        for proof_dict in doc_data_list:
            if proof_dict == None:
                continue
            for text in proof_dict.values():
                all_proofs.add(
                    su.quest_norm(su.normalize(dtk, text))
                )
        proof_data_by_answer.append(list(all_proofs))
    return proof_data_by_answer
