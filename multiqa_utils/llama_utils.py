import glob
import os
import logging

import torch
from transformers import AutoTokenizer, pipeline

from utils.util_classes import Metrics
import utils.file_utils as fu
import utils.run_utils as ru
import multiqa_utils.data_utils as du


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
        

# -------- Colab Utils ---------- #
def prepare_data_for_drive(cfg, dataset, split):
    assert dataset == 'romqa' and split == 'dev'
    in_data = du.get_data(cfg, f'{dataset}_{split}')
    fu.dumppkl(in_data, f'{cfg.data_download_dir}{dataset}/{split}_input_data.pkl')
    

