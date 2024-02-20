import logging
import os

import utils.file_utils as fu
from genre.fairseq_model import GENRE
from genre.entity_linking import (
    get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn,
)
from genre.utils import (
    get_entity_spans_pre_processing,
    get_entity_spans_post_processing,
    get_entity_spans_finalize,
)
from genre.trie import Trie


def load_genre_model_and_data(
    genre_model_path, cand_trie_path, batch_size_toks=2048 * 7
):
    genre_model = GENRE.from_pretrained(genre_model_path).eval()
    genre_model = genre_model.cuda()
    logging.info(">> Model loaded")
    genre_model.cfg.dataset.max_tokens = batch_size_toks
    cand_trie = build_cand_trie(cand_trie_path)
    logging.info(">> Trie loaded")
    return genre_model, cand_trie


# Assumes a list of dicts each containing a "title" and "content" key
def load_and_prepare_wiki_data(file_path):
    file_data = fu.load_file(file_path)
    # TODO: add an id in here somewhere for regrouping
    prefixes = [f'Title: {f["title"]} Text: ' for f in file_data]
    # Brackets break the model, replace with spaces to avoid shifting inds
    text_strs = [f'{p}{f["content"]}'.replace('[', ' ').replace(']', ' ') for p, f in zip(prefixes, file_data)]
    prefix_lens = [len(p) for p in prefixes]
    input_data = [{'text': t, 'pref_len': pl} for t, pl in zip(text_strs, prefix_lens)]
    return input_data


# Assumes input_data is a list of dicts that contain a "text" key
def predict_batch(input_data, genre_model, cand_trie):
    input_sents = [d['text'] for d in input_data]
    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        genre_model,
        input_sents,
        candidates_trie=cand_trie,
    )
    preds = genre_model.sample(
        get_entity_spans_pre_processing(input_sents),
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    return preds


# This is the CPU part
def process_preds_batch(input_data, preds):
    input_sents = [d['text'] for d in input_data]
    all_out_ents = []
    all_out_sents = []
    for i in range(len(preds[0])):
        output_sents = get_entity_spans_post_processing(
            [e[i]["text"] for e in preds]  # this is where we (could) take the best scoring
        )
        out_ents = get_entity_spans_finalize(
            input_sents,
            output_sents,
        )
        all_out_ents.append(out_ents)
        all_out_sents.append(output_sents)

    all_out = []
    for pred_num, text_ents in enumerate(all_out_ents):
        for text_num, ents in enumerate(text_ents):
            if len(all_out) <= text_num:
                all_out.append({
                    'pred': [],
                    'ents': [],
                    'all_ents': set(),
                })
            ents_f = [(s, l, e.replace('_', ' ').strip()) for s, l, e in ents if l > 2]
            all_out[text_num]['pred'].append(preds[text_num][pred_num])
            all_out[text_num]['ents'].append(ents_f)
    for i, ao in enumerate(all_out):
        for es in ao['ents']:
            all_out[i]['all_ents'].update(es)
    return all_out


# This should be done before running any scripts
def build_cand_trie(path='cand_trie.pkl', genre_model=None):
    # Rebuilding takes 20m, loading takes 20s, 20G mem
    if os.path.exists(path):
        return fu.load_file(path)
    assert False, "not fixed yet"

    blink_ents_data = fu.load_file('/scratch/ddr8143/repos/BLINK/models/entity.jsonl')
    all_ents_strs = [d['entity'] for d in blink_ents_data if len(d['text']) != 0]
    cand_trie = Trie(
        [genre_model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in all_ents_strs]
    )
    fu.dumppkl(cand_trie, path)
    return cand_trie
