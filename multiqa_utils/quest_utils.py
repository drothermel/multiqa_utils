# ---- ID ---- #


def get_id(qdata):
    return qdata["id"]


# ---- Question ---- #


def get_question(qdata):
    return qdata["query"]


# ---- Answers ---- #
def get_answer_set(qdata):
    return set([a for a in qdata["docs"]])


def get_answer_dict(qdata):
    return {a: [a] for a in qdata["docs"]}


# ---- Entities ---- #
def get_gtentities(elem):
    return {}
