import glob
import os
import logging

import torch
from transformers import AutoTokenizer, pipeline

from utils.util_classes import Metrics
import utils.file_utils as fu
import utils.run_utils as ru
import multiqa_utils.data_utils as du


def get_dist_out_path(cfg, dataset, split, shard_ind, version):
    out_dir = f'{cfg.download_dir}data/{dataset}/llama2_7b_output/{split}/'
    os.makedirs(out_dir, exist_ok=True)
    return f'{out_dir}data_pt{shard_ind}_{version}.pkl'


def prompt_iter(prompt_base, dataset):
    for qdata in dataset:
        question = qdata['question']
        yield f'{prompt_base}\n\nQuestion: {question}\nAnswer Type:'

def get_model_name(llm_type):
    model_map = {
        "llama2_7b": "meta-llama/Llama-2-7b-hf",
    }
    return model_map[llm_type]

def make_tokenizer_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    llama_pipeline = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        batch_size=1,
    )
    return tokenizer, llama_pipeline

def run_pipeline(
    pipeline,
    tokenizer,
    prompt_base,
    dataset,
):
    logging.info(f">> Running llama2 on data: {len(dataset):,}")
    preds = []
    for i, out in enumerate(pipeline(
        prompt_iter(prompt_base, dataset),
        do_sample=False,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
    )):
        ru.processed_log(i, input_size=len(dataset))
        preds.append(out[0]['generated_text'])
    return preds


def get_lm_pred_from_output_prompt(output, prompt):
    plen = len(prompt)
    assert output[:plen] == prompt
    pred = output[plen:]
    pred_lines = []
    for pl in pred.split('\n'):
        if pl == '':
            continue
        pred_lines.append(pl)
    return pred_lines

def parse_prompt_pred(prompt_type, pred_lines):
    assert prompt_type in ['el_prompt']
    if prompt_type == 'el_prompt':
        keys = {
            'question': 'Question: ',
            'answer_type': 'Answer Type: ',
            'pages': 'Pages: ',
        }
        # Get the first instance of this key if exists
        out = {}
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
    
def link_outputs_to_data_ids(prompt_t, prompt, in_data, out_data):
    metrics = Metrics()
    
    # Map input data by question
    q2data = {qdata['question']: qdata for qdata in in_data}
    
    # Map question to outputs
    q2outs = {}
    for odata in out_data:
        lm_pred = get_lm_pred_from_output_prompt(odata, prompt)
        parsed = parse_prompt_pred(prompt_t, lm_pred)
        
        # Collect stats on parsed data
        qinfo = {'pred': lm_pred, 'parsed': parsed}
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

def load_parse_match_prompt_pred_data(
    cfg,
    data_type,
    llm,
    prompt_type,
    in_data=None,
    out_data=None,
    prompt=None,
):
    # Load data if necessary
    dataset, split = du.data_type_to_name_split(data_type)
    if in_data is None:
        in_data = du.get_data(cfg, data_type)
    if out_data is None:
        out_data = load_llama2_output_data(cfg, dataset, split)
    if prompt is None:
        prompt = fu.load_file(du.get_prompt_path(cfg, llm, prompt_type, dataset))
    
    id2outs, metrics = link_outputs_to_data_ids(
        prompt_type, prompt, in_data, out_data,
    )
    missing_inds = get_missing_question_inds(
        in_data,
        metrics['unparsed_qs__data'],
    )
    return id2outs, metrics, missing_inds
        
        

# -------- Colab Utils ---------- #
def prepare_data_for_drive(cfg, dataset, split):
    assert dataset == 'romqa' and split == 'dev'
    in_data = du.get_data(cfg, f'{dataset}_{split}')
    fu.dumppkl(in_data, f'{cfg.data_download_dir}{dataset}/{split}_input_data.pkl')
    

def load_llama2_output_data(cfg, dataset, split):
    data_dir = f'{cfg.data_download_dir}{dataset}/llama2_7b_output/{split}/'
    version = cfg.entity_linking.prompt_versions[dataset]
    all_files = glob.glob(f'{data_dir}/data_pt*_{version}.pkl')
    all_lines = []
    for f in all_files:
        all_lines.extend(fu.load_file(f))
    return all_lines
