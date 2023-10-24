import copy
import glob

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


def dataname_to_shortname(data_name):
    all_shortnames = set([v[0] for v in DATASET_NAMES.values()])
    if data_name in all_shortnames:
        return data_name
    if data_name in DATASET_NAMES:
        return DATASET_NAMES[data_name][0]
    assert False, f">> No such dataset: {data_name}"


def get_base_data_path(cfg, dataset, split, raw=False):
    if raw:
        filename = cfg.datasets.orig_data_filename[dataset][split]
        data_dir = cfg.data_download_dir
        return f"{data_dir}{dataset}/{filename}"
    else:
        data_dir = cfg.postp_data_dir
        return f"{data_dir}{dataset}_{split}_data.jsonl"


def get_linked_data_path(cfg, dataset, split, link_type, raw=False):
    if raw:
        if link_type in ['elq_ent', 'elq_ori_str']:
            short_name = dataname_to_shortname(dataset)
            data_dirname_base = cfg.entity_linking.elq_raw_dirname_base
            data_dirname = f"{data_dirname_base}__{short_name}_{split}/"
            return f"{cfg.postp_data_dir}{data_dirname}biencoder_outs.jsonl"
        else:
            assert False, f">> Linked type not implemented: {link_type}"
    else:
        data_dir = cfg.postp_data_dir
        linked_vs = cfg.entity_linking.linked_versions
        assert link_type in linked_vs, f">> ERROR: {link_type} doesn't exist"
        assert dataset in linked_vs[link_type], f'>> {dataset} not in linked_vs'
        assert split in linked_vs[link_type][dataset], f'>> {split} not in linked_vs'
        version = cfg.entity_linking.linked_versions[link_type][dataset][split]
        return f"{data_dir}{link_type}_{dataset}_{split}_{version}.jsonl"


def get_prompt_path(cfg, lm, prompt_type, dataset):
    if prompt_type in ["el_prompt"]:
        version = cfg.entity_linking.prompt_versions.get(dataset, None)
        if version is None:
            assert False, f">> ERROR: No prompt for {prompt_type} {lm} {dataset}."
        prompt_dir = cfg.prompt_dir
        return f"{prompt_dir}{lm}_{prompt_type}_{dataset}_{version}.txt"
    else:
        assert False, "Unimplemented yet"


# ---------- Dataset Loading ----------- #

def get_data_type(data_name, split):
    return f'{data_name}_{split}'

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


def get_data(
    cfg,
    data_type,
    link_type=None,
    raw=False,
    as_dict=False,
    verbose=False,
):
    dataset_name, split = data_type_to_name_split(data_type)
    if link_type is None:
        path = get_base_data_path(cfg, dataset_name, split, raw=raw)
    else:
        path = get_linked_data_path(cfg, dataset_name, split, link_type, raw=raw)

    data = fu.load_file(path, verbose=verbose)
    if as_dict and not raw:
        return {d["id"]: d for d in data}
    elif as_dict and raw:
        print(">> Cannot make raw data into a dict, returning a list")
    return data


# ---------- Orig Dataset Access Utils ----------- #


def visualize_datasets(datasets, ind, max_len=100, dtk=None):
    if dtk is None:
        dtk = su.get_detokenizer()

    for data_name, data_dev in datasets.items():
        print(f"{data_name:20}")
        print(f'  - {"num_elems:":20} {len(data_dev)}')
        print(f'  - {"id:":20} {get_original_id(data_dev[ind], data_name)}')
        print(
            f'  - {"question_type:":20} {get_question_type(data_dev[ind], data_name)}'
        )
        print(f'  - {"question:":20} {get_question(data_dev[ind], data_name)}')
        ans_sets = str(get_answer_sets(data_dev[ind], data_name))
        print(
            f'  - {"answer_sets:":20} {ans_sets[:max_len]}'
        )
        print(f'  - {"gt_ent_sets:":20} {get_gt_ent_sets(data_dev[ind], data_name)}')
        proof_data = str(get_proof_data(data_dev[ind], dtk, data_name))
        print(
            f'  - {"proof_data:":20} {proof_data[:max_len]}'
        )
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
        new_data["original_id"] = get_original_id(qdata, data_name)
        new_data["id"] = f"{qind}__{data_name}__{split}"
        new_data["question"] = get_question(qdata, data_name)
        new_data["question_type"] = get_question_type(qdata, data_name)
        new_data["answer_sets"] = get_answer_sets(qdata, data_name)
        new_data["gt_ent_sets"] = get_gt_ent_sets(qdata, data_name)
        new_data["proof_data"] = get_proof_data(qdata, dtk, data_name)
        preprocessed_data.append(new_data)

    # == Any special fxns for a dataset == #

    # The main version of the data has a train and dev split but doesn't
    # contain all the annotations so add them back in to the preprocessed
    # version and get the gt_entities
    if data_name in DATASET_NAMES["webqsp"]:
        # Includes the train and dev split
        base_webqsp_data = fu.load_file(extra_data_paths["webqsp"]["base_train"])
        matched_data = wsp.match_rectify_to_original(
            preprocessed_data,
            base_webqsp_data,
        )
        for i, md in enumerate(matched_data):
            preprocessed_data[i]["gt_ent_sets"] = wsp.get_gt_ent_sets(md)
    return preprocessed_data


# ---- Entity Linking Data Utils ---- #

# == ELQ == #
def get_elq_ents(edata_elem):
    return [ent for ent, ori in edata_elem["pred_tuples_string"]]


def get_elq_ori_strs(edata_elem):
    return [ori for ent, ori in edata_elem["pred_tuples_string"]]


def get_elq_ent_oris(edata_elem):
    return {
        'elq_ent': get_elq_ents(edata_elem),
        'elq_ori_str': get_elq_ori_strs(edata_elem),
    }

# == Llama2 == #
def get_llama2_dist_out_dir(cfg, dataset, split):
    return f'{cfg.download_dir}data/{dataset}/llama2_7b_output/{split}/'

def get_llama2_dist_out_path(cfg, dataset, split, shard_ind, version):
    out_dir = get_llama2_dist_out_dir(cfg, dataset, split)
    os.makedirs(out_dir, exist_ok=True)
    return f'{out_dir}data_pt{shard_ind}_{version}.pkl'

"""
# Current method of interacting with this:
llama2_raw_data = du.load_llama2_output_data(cfg, dataset, split)
prompt = fu.load_file(du.get_prompt_path(cfg, 'llama2_7b', 'el_prompt', dataset))
du.parse_llama2_prompt_pred('el_prompt', llama2_raw_data[0], prompt)
"""

# TODO: add this to get_data instead of having standalone
def load_llama2_output_data(cfg, dataset, split):
    data_dir = get_llama2_dist_out_dir(cfg, dataset, split)
    version = cfg.entity_linking.prompt_versions[dataset]
    all_files = glob.glob(f'{data_dir}/data_pt*_{version}.pkl')
    all_lines = []
    for f in all_files:
        all_lines.extend(fu.load_file(f, verbose=False))
    return all_lines

def get_llama2_pred_from_output_prompt(output, prompt):
    plen = len(prompt)
    assert output[:plen] == prompt
    pred = output[plen:]
    pred_lines = []
    for pl in pred.split('\n'):
        if pl == '':
            continue
        pred_lines.append(pl)
    return pred, pred_lines

def parse_llama2_prompt_pred(prompt_type, output, prompt):
    full_pred, pred_lines = get_llama2_pred_from_output_prompt(output, prompt)
    out = {
        'full_pred': full_pred,
        'pred_lines': pred_lines,
    }
    assert prompt_type in ['el_prompt']
    if prompt_type == 'el_prompt':
        keys = {
            'question': 'Question: ',
            'answer_type': 'Answer Type: ',
            'pages': 'Pages: ',
        }
        # Get the first instance of this key if exists
        for key, key_str in keys.items():
            for l in pred_lines:
                if key_str in l:
                    out[key] = l[len(key_str):]
                    break

        if 'question' not in out:
            return {}
        for k in ['answer_type', 'pages']:
            if k in out:
                out[k] = list(set([v.strip() for v in out[k].split(',')]))
        return out
    
# TODO: move this to the verify script!
def link_outputs_to_data_ids(prompt_t, prompt, in_data, out_data):
    metrics = Metrics()
    
    # Map input data by question
    q2data = {qdata['question']: qdata for qdata in in_data}
    
    # Map question to outputs
    q2outs = {}
    for odata in out_data:
        parsed = parse_llama2_prompt_pred(prompt_t, odata, prompt)
        
        # Collect stats on parsed data
        qinfo = {
            'pred': parsed['pred_lines'],
            'parsed': parsed,
        }
        for k in ['question', 'answer_type', 'pages']:
            if k not in parsed:
                metrics.increment_val(name=f'{k}_notin_lmout', amount=1)
                metrics.add_example(name=f'{k}_notin_lmout__data', item=qinfo)
        if 'question' not in parsed:
            continue
            
        # Verify that the parsed question is in the input dataset
        if parsed['question'] not in q2data:
            metrics.increment_val(name='lmq_notin_indata', amount=1)
            metrics.add_example(name='lmq_notin_indata__data', item=qinfo)
            continue
        q2outs[parsed['question']] = parsed
    
    # identify questions from in_data missing int output
    missing_qs = set(q2data.keys()) - set(q2outs.keys())
    metrics.increment_val(name='unparsed_qs', amount=len(missing_qs))
    metrics.add_examples(name='unparsed_qs__data', items=missing_qs)
    
    # Map from q2outs to id2outs
    id2outs = {q2data[q]['id']: out for q, out in q2outs.items()}
    return id2outs, metrics.to_dict()

def get_missing_question_inds(in_data, missing_qlist):
    missing_inds = []
    missing_set = set(missing_qlist)
    for i, qdata in enumerate(in_data):
        if qdata['question'] in missing_set:
            missing_inds.append(i)
    return missing_inds


if __name__ == "__main__":
    print("Loading")
    qmp_dev_path = "/scratch/ddr8143/multiqa/downloads/data/qampari/dev_data.jsonl"
    qmp_dev = fu.load_file(qmp_dev_path)

    print("Running")
    test_qd = [qd for i, qd in enumerate(qmp_dev) if i < 10]
    # proof_data = qmp_raw_to_proof_info(test_qd)
    # proof_query_list = qmp_proof_data_to_query_list(proof_data)
    # breakpoint()
