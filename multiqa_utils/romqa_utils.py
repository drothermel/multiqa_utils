# # =============================================== # #
# # =============== Info Extractors =============== # #
# # =============================================== # #

# --------- Original Data Format Extractors ---------- #


def get_original_id(qdata):
    return qdata['id']


def get_question(qdata):
    return qdata['question']


def get_question_type(qdata):
    return 'complex_multi'


# Notes:
# - 51/7068 questions in RQA dev set have no answers. Explains why eval
#   calcs recall as: len(common) / max(1, len(gold)).  These q's will
#   always be p = r = f1 = 0.
# - We can get the urls from the URIs if we want to by:
# for qid in Q1061264 Q1061264 Q1061264; do
#    curl https://hub.toolforge.org/${qid}\?lang\=en >> output.txt;
#    echo " ${qid}" >> output.txt;
# done
# - But even more clean would be to use the TRex data that they used
#   to construct the dataset: https://hadyelsahar.github.io/t-rex/
# - ah, but in run_all_baselines:
#   data = [ex for ex in data if ex['complete_answer']]
#   so they dump the bad data points when reporting results
def get_answer_sets(qdata):
    all_ans = []
    for adata in qdata['complete_answer']:
        all_ans.append(list(set([adata['text'], *adata['aliases']])))
    possible_answer_sets = [all_ans]
    return possible_answer_sets


def get_gt_ent_sets(qdata):
    ent_sets = set()
    for ed in qdata['constraints']:
        o_ent = ed['other_ent']
        ent_sets.add(frozenset([o_ent['text'], *o_ent['aliases']]))
    return [list(fs) for fs in ent_sets]


# Notes: Built from wikidata graph & retrieval from T-Rex not wikipedia, have
#        dpr scored passages but not the correct passages.
def get_proof_data(qdata, dtk):
    return None
