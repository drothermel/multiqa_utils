import copy

import utils.file_utils as fu
import multiqa_utils.qampari_utils as qmp
import multiqa_utils.romqa_utils as rqa
import multiqa_utils.quest_utils as qst
import multiqa_utils.ambigqa_utils as amb
import multiqa_utils.webqsp_utils as wsp
import multiqa_utils.cwq_utils as cwq
import multiqa_utils.string_utils as su

# ---------- Dataset Names and Paths ----------- #

# Order matters, do not reorder!
DATASET_UTILS = {
    "qmp": qmp,
    "qampari": qmp,
    "rqa": rqa,
    "romqa": rqa,
    "qst": qst,
    "quest": qst,
    "amb": amb,
    "ambigqa": amb,
    "wsp": wsp,
    "webqsp": wsp,
    "cwq": cwq,
}
DATASET_NAMES = {
    "qampari": ("qmp", "qampari"),
    "romqa": ("rqa", "romqa"),
    "quest": ("qst", "quest"),
    "ambigqa": ("amb", "ambigqa"),
    "webqsp": ("wsp", "webqsp"),
    "cwq": ("cwq", "cwq"),
}
SPLIT_NAMES = ["dev", "train", "test"]


def get_base_data_path(cfg, dataset, split):
    data_dir = cfg.postp_data_dir
    return f'{data_dir}{dataset}_{split}_data.jsonl'

        
def get_orig_data_path(cfg, dataset, split):
    filename = cfg.datasets.orig_data_filename[dataset][split]
    data_dir = cfg.data_download_dir
    return f'{data_dir}{dataset}/{filename}'

    
def get_linked_data_path(cfg, link_type, dataset, split):
    data_dir = cfg.entity_linking.postp_data_dir
    linked_v = cfg.entity_linking.linked_versions
    assert link_t in linked_v, f">> ERROR: {link_t} doesn't exist"
    if dataset not in linked_v[link_t] or split not in linked_v[link_t][dataset]:
        return None
    version = el_cfg.linked_versions[link_t][dataset][split]
    return f'{data_dir}{link_t}_{dataset}_{split}_{version}.jsonl'


def get_prompt_path(cfg, lm, prompt_type, dataset):
    if prompt_type in ['el_prompt']:
        version = cfg.entity_linking.prompt_versions.get(dataset, None)
        if version is None:
            assert False, f">> ERROR: No prompt for {prompt_type} {lm} {dataset}."
        prompt_dir = cfg.prompt_dir
        return f'{prompt_dir}{lm}_{prompt_type}_{dataset}_{version}.txt'
    else:
        assert False, 'Unimplemented yet'

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

def load_linked_data(el_cfg, data_t, link_t):
    dataset, split = data_type_to_name_split(data_t)
    data_dir = el_cfg.postp_data_dir
    assert link_t in el_cfg.linked_versions
    version = el_cfg.linked_versions[link_t][dataset][split]
    path = f'{data_dir}{link_t}_{dataset}_{split}_{version}.jsonl'
    return fu.load_file(path)
    
def load_prompt(lm, prompt_type, dataset, version='v0'):
    assert lm in ['llama2_7b']
    assert prompt_type in ['elq_prompt']
    prompt_dir = cfg.entity_linking.prompt_dir
    version = cfg.entity_linking.prompt_versions[dataset]
    path = f'{prompt_dir}{lm}_{prompt_type}_{dataset}_{version}.txt'
    prompt = fu.load_file(path)


def get_data(
    cfg,
    data_type,
    linked_type=None,
    orig_data=False,
    as_dict=False,
    verbose=False,
):
    dataset_name, split = data_type_to_name_split(data_type)
    if linked_type is not None:
        path = get_linked_data_path(cfg, dataset_name, split)
    elif orig_data:
        path = get_orig_data_path(cfg, dataset_name, split)
    else:
        path = get_base_data_path(cfg, dataset_name, split)
        
    data = fu.load_file(path, verbose=verbose)
    if as_dict and not orig_data:
        return {d['id']: d for d in data}
    elif as_dict and orig_data:
        print(f">> Cannot make orig_data into a dict, returning a list")
    return data


# ---------- Orig Dataset Access Utils ----------- #

def visualize_datasets(datasets, ind, max_len=100, dtk=None):
    if dtk is None:
        dtk = su.get_detokenizer()
        
    for data_name, data_dev in datasets.items():
        print(f'{data_name:20}')
        print(f'  - {"num_elems:":20} {len(data_dev)}')
        print(f'  - {"id:":20} {get_original_id(data_dev[ind], data_name)}')
        print(f'  - {"question_type:":20} {get_question_type(data_dev[ind], data_name)}')
        print(f'  - {"question:":20} {get_question(data_dev[ind], data_name)}')
        print(f'  - {"answer_sets:":20} {str(get_answer_sets(data_dev[ind], data_name))[:max_len]}')
        print(f'  - {"gt_ent_sets:":20} {get_gt_ent_sets(data_dev[ind], data_name)}')
        print(f'  - {"proof_data:":20} {str(get_proof_data(data_dev[ind], dtk, data_name))[:max_len]}')
        print()

def get_original_id(data_elem, data_name):
    return DATASET_UTILS[data_name].get_original_id(data_elem)


def get_question(data_elem, data_name):
    return DATASET_UTILS[data_name].get_question(data_elem)

def get_question_type(data_elem, data_name):
    return DATASET_UTILS[data_name].get_question_type(data_elem)


def get_answer_sets(data_elem, data_name):
    return DATASET_UTILS[data_name].get_answer_sets(data_elem)


def get_gt_ent_sets(data_elem, data_name):
    return DATASET_UTILS[data_name].get_gt_ent_sets(data_elem)

def get_proof_data(data_elem, dtk, data_name):
    return DATASET_UTILS[data_name].get_proof_data(data_elem, dtk)


# ---------- Dataset Preprocessing Helpers ----------- #

def preprocess_dataset(data_name, split, orig_data, extra_data_paths):
    dtk = su.get_detokenizer()
    preprocessed_data = []
    for qind, qdata in enumerate(orig_data):
        new_data = copy.deepcopy(qdata)
        new_data['original_id'] = get_original_id(qdata, data_name)
        new_data['id'] = f'{qind}__{data_name}__{split}'
        new_data['question'] = get_question(qdata, data_name)
        new_data['question_type'] = get_question_type(qdata, data_name)
        new_data['answer_sets'] = get_answer_sets(qdata, data_name)
        new_data['gt_ent_sets'] = get_gt_ent_sets(qdata, data_name)
        new_data['proof_data'] = get_proof_data(qdata, dtk, data_name)
        preprocessed_data.append(new_data)

    # == Any special fxns for a dataset == #

    # The main version of the data has a train and dev split but doesn't
    # contain all the annotations so add them back in to the preprocessed
    # version and get the gt_entities
    if data_name in DATASET_NAMES['webqsp']:
        # Includes the train and dev split
        base_webqsp_data = fu.load_file(
            extra_data_paths['webqsp']['base_train']
        )
        matched_data = wsp.match_rectify_to_original(
            preprocessed_data,
            base_webqsp_data,
        )
        for i, md in enumerate(matched_data):
            preprocessed_data[i]['gt_ent_sets'] = wsp.get_gt_ent_sets(md)
    return preprocessed_data

# ---- Entity Linking Metrics ---- #

# Parse elq data consistently (no accidental flips!)
def get_elq_ents(edata_elem):
    return set([ent for ent, ori in edata['pred_tuples_string']])

def get_elq_oris(edata_elem):
    return set([ori for ent, ori in edata['pred_tuples_string']])

def get_elq_ent_oris(edata_elem):
    return [{'ent': ent, 'ori': ori} for ent, ori in edata['pred_tuples_string']]
    

if __name__ == "__main__":
    print("Loading")
    qmp_dev_path = "/scratch/ddr8143/multiqa/downloads/data/qampari/dev_data.jsonl"
    qmp_dev = fu.load_file(qmp_dev_path)

    print("Running")
    test_qd = [qd for i, qd in enumerate(qmp_dev) if i < 10]
    # proof_data = qmp_raw_to_proof_info(test_qd)
    # proof_query_list = qmp_proof_data_to_query_list(proof_data)
    # breakpoint()
