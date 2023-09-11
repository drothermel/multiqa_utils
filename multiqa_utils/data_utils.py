import utils.file_utils as fu
import multiqa_utils.qampari_utils as qmp
import multiqa_utils.romqa_utils as rqa
import multiqa_utils.quest_utils as qst

# Order matters, do not reorder!
DATASET_UTILS = {
    "qmp": qmp,
    "qampari": qmp,
    "rqa": rqa,
    "romqa": rqa,
    "qst": qst,
    "quest": qst,
}
DATASET_NAMES = {
    "qampari": ("qmp", "qampari"),
    "romqa": ("rqa", "romqa"),
    "quest": ("qst", "quest"),
}
SPLIT_NAMES = ["dev", "train", "test"]

# ---------- Dataset Loading ----------- #


def data_type_to_name_split(data_type):
    splits = []
    names = []
    for split in SPLIT_NAMES:
        if split in data_type:
            splits.append(split)

    for name, aliases in DATASET_NAMES.items():
        for a in aliases:
            if a in data_type:
                names.append(name)
                break

    assert len(splits) == 1 and len(names) == 1, f"Invalid type {data_type}"
    return names[0], splits[0]


def data_to_dict(data_list, data_type):
    return {get_id(d, name): d for d in data_list}


# path_cfg: must contain elq_datasets and base_datasets
def get_data(path_cfg, data_type, as_dict=False, verbose=False):
    name, split = data_type_to_name_split(data_type)
    base_data = fu.load_file(path_cfg[name][split], verbose=verbose)
    if as_dict:
        if "elq" in data_type:
            return {d["id"]: d for d in base_data}
        else:
            return data_to_dict(base_data, data_type)
    return base_data


# ---------- Dataset Access Utils ----------- #


def get_id(data_elem, data_name):
    return DATASET_UTILS[data_name].get_id(data_elem)


def get_question(data_elem, data_name):
    return DATASET_UTILS[data_name].get_question(data_elem)


def get_answer_set(data_elem, data_name):
    return DATASET_UTILS[data_name].get_answer_set(data_elem)


def get_answer_dict(data_elem, data_name):
    return DATASET_UTILS[data_name].get_answer_dict(data_elem)


def get_gtentities(data_elem, data_name):
    return DATASET_UTILS[data_name].get_gtentities(data_elem)


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


if __name__ == "__main__":
    print("Loading")
    qmp_dev_path = "/scratch/ddr8143/multiqa/downloads/data/qampari/dev_data.jsonl"
    qmp_dev = fu.load_file(qmp_dev_path)

    print("Running")
    test_qd = [qd for i, qd in enumerate(qmp_dev) if i < 10]
    # proof_data = qmp_raw_to_proof_info(test_qd)
    # proof_query_list = qmp_proof_data_to_query_list(proof_data)
    # breakpoint()
