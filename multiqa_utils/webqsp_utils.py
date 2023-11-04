# # =============================================== # #
# # =============== Info Extractors =============== # #
# # =============================================== # #


def match_rectify_to_original(
    rectify_original_data,
    dataset_original_data,
):
    orig_dict = {od['QuestionId']: od for od in dataset_original_data['Questions']}
    updated = []
    for rod in rectify_original_data:
        od = orig_dict[rod['original_id']]
        new_dict = {**rod, **od}
        updated.append(new_dict)
    return updated


# --------- Original Data Format Extractors ---------- #


def get_original_id(qdata):
    return qdata['id']


def get_question(qdata):
    return qdata['question']


def get_question_type(qdata):
    if len(qdata['answers']) > 1:
        return 'simple_multi'
    else:
        return 'simple_single'


# Notes: orig had multiple possible answer sets but rectify has one
def get_answer_sets(qdata):
    ann_ans = [list(set([a])) for a in qdata['answers']]
    possible_answer_sets = [ann_ans]
    return possible_answer_sets


# Notes: each parse has only one TopicEntityName
# this is the only one that acts on the true original data not on
# the rectify version because they didn't include this parse data.
def get_gt_ent_sets(qdata):
    if "Parses" not in qdata:
        return None
    ent_sets = [set()]
    for parse in qdata['Parses']:
        ent_sets[0].add(parse['TopicEntityName'])
    return [list(es) for es in ent_sets]


# Notes: None, original retrieval from web broadly
def get_proof_data(qdata, dtk):
    return None
