import os
import json
import jsonlines
import urllib
import unicodedata
import shutil

def current_default_path_args():
    path_config = {
        # Input Data from previous steps
        'gpt_ans_path': '/scratch/ddr8143/multiqa/qampari_data/qmp_simple_gpt3_answers_structured.json',
        'elq_ans_path': '/scratch/ddr8143/multiqa/qampari_data/eql_default_tagging_v0_qmp_dev.jsonl',
        # Key Directories
        'gt_wiki_dir': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/',
        'gt_wiki_postp_dir': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/',
        # Paths for setting up ent_str to wiki_page cache
        'gt_title_set_path': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/wikiversion_title_set.json',
        'wikitags_path_regexp': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/title2linktagmestrs_*.json',
        'strs_for_cache_path': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/strs_to_add_to_cache_v0.json',
        'cache_path': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_cache.json',
        'disambig_cache_path': '/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_disambig_cache.json',
    }
    path_args = argparse.Namespace(**path_config)
    return path_args

def loadjsonl(filename):
    all_lines = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            all_lines.append(obj)
    return all_lines


def normalize(text, unquote=False):
    if unquote:
        text = urllib.parse.unquote(text)
    text = unicodedata.normalize('NFD', text).lower().replace(' ', '_')
    return text


def unnormalize(text):
    return text.replace('_', ' ')

def get_backup_path(path):
    return f"{path}__old"

def checkpoint_json(
    data,
    path,
    suffix='',
    backup=True,
):
    cp_base = f"{path}{suffix}"
    if backup:
        cp_backup = get_backup_path(cp_base)
        if os.path.exists(cp_base):
            shutil.move(cp_base, cp_backup)
    json.dump(data, open(cp_base, 'w+'))
    print(">> Dumped:", cp_base)