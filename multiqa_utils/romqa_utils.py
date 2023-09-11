
# ---- ID ---- #

def get_id(qdata):
    return qdata['id']

# ---- Question ---- #

def get_question(qdata):
    return qdata['question']

# ---- Answers ---- #
def get_answer_set(qdata):
    return set([a["text"] for a in d["complete_answer"]])

def get_answer_dict(qdata):
    return {a["text"]: a["aliases"] for a in d["complete_answer"]}

# ---- Entities ---- #

def get_gtentities(elem):
    # Note that rqa entities are always good due to dataset construction
    ent2urlalias = {}
    for c in elem["constraints"]:
        ent = c["other_ent"]["text"]
        if ent not in ent2urlalias:
            ent2urlalias[ent] = {"url": c["other_ent"]["uri"], "aliases": set()}
        ent2urlalias[ent]["aliases"].update(c["other_ent"]["aliases"])
    return ent2urlalias
