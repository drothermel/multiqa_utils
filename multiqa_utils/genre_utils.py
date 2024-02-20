import utils.file_utils as fu
from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import (
	get_entity_spans_pre_processing,
	get_entity_spans_post_processing,
	get_entity_spans_finalize
)
from genre.trie import Trie


def load_genre_model_and_data(genre_model_path, ent_trie_path, batch_size_toks=2048*7):
	model = GENRE.from_pretrained(genre_model_path).eval()
	model = model.cuda()
	logging.info(">> Model loaded")
	model.cfg.dataset.max_tokens = batch_size_toks
	ent_cands_trie = build_cand_trie(ent_trie_path)
	logging.info(">> Trie loaded")
    return model, ent_cands_trie

# Assumes a list of dicts each containing a "title" and "content" key
def load_and_prepare_wiki_data(file_path):
    file_data = fu.load_file(file_path)
    prefixes = [f'Title: {f["title"]} Text:' for f in file_data]
    text_strs = [
        f'{p}{f["content"]}' for p, f in zip(prefixes, file_data)
    ]
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
def process_preds_batch(input_sents, preds):
    output_sentences = get_entity_spans_post_processing(
        [e[0]["text"] for e in preds] # this is where we take the best scoring
    )
    finalized_out = get_entity_spans_finalize(
        input_sents, output_sentences,
    )
    finalized_out_filtered = [
        [(s, l, e.replace('_', ' ').strip()) for s, l, e in fo if l > 2] for fo in finalized_out
    ]
    all_out = [{
        'input': input_sents[i],
        'output': output_sentences[i],
        'score': preds[i][0]['score'].item(),
        'entities': finalized_out[i],
        'entities_filtered': finalized_out_filtered,
    } for i in range(len(input_sents))]
    return all_out


# This should be done before running any scripts
def build_cand_trie(path='ent_cands_trie.pkl'):
    # Rebuilding takes 20m, loading takes 20s, 20G mem
    if os.path.exists(path):
        return fu.load_file(path)
    
    blink_ents_data = fu.load_file('/scratch/ddr8143/repos/BLINK/models/entity.jsonl')
    all_ents_strs = [d['entity'] for d in blink_ents_data if len(d['text']) != 0]
    ent_cands_trie = Trie([
        model.encode(" }} [ {} ]".format(e))[1:].tolist()
        for e in all_ents_strs
    ])
    fu.dumppkl(ent_cands_trie, path)
    return ent_cands_trie
