import json
import os
import tqdm

import multiqa_utils.general_utils as gu


###################################
##         Input Parsing         ##
###################################

def gpt_out_to_info(gout):
    ans_lines = gout['output'].split('\n')
    gpt_info = {'qid': gout['qid']}
    for l in ans_lines:
        sl = l.split(": ")
        a_key = gu.normalize(sl[0])
        if "pages" in a_key:
            a_key = "pages"
        elif "answers" in a_key:
            a_key = "sampled_answers"
        elif "type" in a_key:
            a_key = "answer_type"
        else:
            print(">> Unknown answer key:", a_key)
            assert False
        a_list = [gu.normalize(a) for a in sl[1].split(", ")]
        gpt_info[a_key] = a_list
    return gpt_info


def convert_gpt_raw_to_structured(raw_path, structured_path, force=False):
    if os.path.exists(structured_path) and not force:
        print(">> Structure GPT out already exists:", structured_path)
        return
    gpt_ans_raw = json.load(open(raw_path))
    gpt_ans = [gpt_out_to_info(ga) for ga in gpt_ans_raw]
    json.dump(gpt_ans, open(structured_path, 'w+'))
    print(">> Dumped structured answers to:", structured_path)

def gpt_structured_to_norm_ent(gpt):
    all_ents = set()
    all_ents.update([gu.normalize(p) for p in gpt['pages']])
    all_ents.update([gu.normalize(a) for a in gpt['sampled_answers']])
    return all_ents

def gpt_structuredlist_to_norm_ent(gpt_list):
    all_ents = set()
    for gpt in gpt_list:
        all_ents.update(gpt_structured_to_norm_ent(gpt))
    return all_ents

def elq_ans_to_unique_norm_ent(elq_ans):
    return set([gu.normalize(e[0]) for e in elq_ans['pred_tuples_string']])

def elq_anslist_to_unique_norm_ent(elq_ans_list):
    unique_elq_str = set()
    for ea in elq_ans_list:
        ea_unique_set = elq_ans_to_unique_norm_ent(ea)
        unique_elq_str.update(ea_unique_set)
    return unique_elq_str

def wikipedia_tags_to_unique_norm_ent(wiki_tags):
    all_ent_strs = set()
    all_ent_strs.update([gu.normalize(s, unquote=True) for s in wiki_tags['links'] if '://' not in s])
    all_ent_strs.update([gu.normalize(s) for s in wiki_tags['tagmes'] if '://' not in s])
    return all_ent_strs
    
    
def wikipedia_tagsfilelist_to_unique_norm_ent(wiki_tags_file_list, use_tqdm=False):
    all_ents = set()
    for f in tqdm.tqdm(wiki_tags_file_list, disable=(not use_tqdm)):
        wtlines = json.load(open(f))
        for _, data in wtlines.items():
            file_ents = wikipedia_tags_to_unique_norm_ent(data)
            all_ents.update(file_ents)
    return all_ents


###################################
##        Pipeline Pieces        ##
###################################

def aggregate_strs_to_add_to_cache(
    path_args,
    add_elq=False,
    add_gpt=False,
    add_wikitags=False,
    use_tqdm=False,
    curr_cache=None
):
    output_path = path_args.strs_for_cache_path
    assert output_path is not None
    
    all_strs = set('')
    if os.path.exists(output_path):
        print(">> Load existing string list:", output_path)
        all_strs = set(json.load(open(output_path)))
        print(">> Initial string list length:", len(all_strs))
        
    # Add elq
    if add_elq and os.path.exists(path_args.elq_ans_path):
        print(">> Adding ELQ ents")
        elq_ans_list = gu.loadjsonl(path_args.elq_ans_path)
        elq_ent_set = elq_anslist_to_unique_norm_ent(elq_ans_list)
        all_strs.update(elq_ent_set)
        print(">> After Adding ELQ:", len(all_strs))
    
    # Add GPT3
    if add_gpt and os.path.exists(path_args.gpt_ans_path):
        print(">> Adding GPT3 ents")
        gpt_ans_list = json.load(open(path_args.gpt_ans_path))
        gpt_ent_set = gpt_structuredlist_to_norm_ent(gpt_ans_list)
        all_strs.update(gpt_ent_set)
        print(">> After Adding GPT3:", len(all_strs))
    
    # Add tagme and links
    if add_wikitags:
        files = glob.glob(path_args.processed_wikitags_path_regexp)
        if len(files) is not None:
            print(">> Adding Wikipedia Tags and Links")
            wt_strs = wikipedia_tagsfilelist_to_unique_norm_ent(files, use_tqdm=use_tqdm)
            all_strs.update(wt_strs)
            print(">> After Adding Wikipedia Tags and Links:", len(all_strs))
            
    if curr_cache is not None:
        print(">> Removing strings already in cache")
        all_strs = all_strs - curr_cache.keys()
        print(">> New string list length:", len(all_strs))
    
    if output_path is not None:
        print(">> Writing file")
        json.dump(list(all_strs - set([''])), open(output_path, 'w+'))
        print(">> Dumped to:", output_path)
    else:
        return all_strs


###################################
##         BM25 Retrieval        ##
###################################

def bm25_out_name(outdir, dataset, split, hits):
    return f"{outdir}/bm25.{dataset}.{split}.h{hits}.json"

# This counts the number of questions in a dataset has no positive included
# (with the assumption that the positive contexts have been produced by some retrieval)
def count_no_positive(dataset_path, hits):
    # In making the datasets I sometimes left the base dataset as only the questions with
    # positive samples in which case the full set is in <filename>__w_no_pos
    no_pos_path = f"{dataset_path}__w_no_pos"
    if os.path.exists(no_pos_path):
        dataset = json.load(open(no_pos_path))
    else:
        dataset = json.load(open(dataset_path))
    total_len = len(dataset)
    no_pos_len = len([d for d in dataset if len(d["positive_ctxs"]) == 0])
    return {"hits": hits, "no_pos_len": no_pos_len, "total_len": total_len}


# And this uses the previous util to display the results
# For example:
# >> ambigqa_bm25_pathname_fxn = lambda hits: ru.bm25_out_name(outdir, "ambigqa_light", "dev", hits)
# >> display_no_positive([100, 400, 1000], ambigqa_bm25_pathname_fxn)
# Hits | No Positive Contexts Retrieved
# ---- | ------------------------------
# 100 | 319/2002 (15.93%)
# 400 | 221/2002 (11.04%)
# 1000 | 170/2002 (8.49%)
#
def display_no_positive(hits_list, pathname_fxn):
    print(f"Hits | No Positive Contexts Retrieved")
    print(f"---- | ------------------------------")
    for hits in hits_list:
        s = count_no_positive(pathname_fxn(hits), hits)
        percent_no_pos = s["no_pos_len"] * 100 / s["total_len"]
        print(
            f"{s['hits']:4} | {s['no_pos_len']}/{s['total_len']} ({percent_no_pos:0.2f}%)"
        )
