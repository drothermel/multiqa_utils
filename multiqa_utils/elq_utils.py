import os
import json
import argparse
import logging
import time

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertTokenizerFast

from elq.index.faiss_indexer import DenseHNSWFlatIndexer
from elq.biencoder.biencoder import load_biencoder

# Note: Separate file bc elq library requires different
# conda environment (el4qa).

# Extracted what I needed from el4qa dense_main because I wasn't
# able to get logging, etc to work by wrapping the script.
# Vast majority of code from:
# https://github.com/facebookresearch/BLINK/tree/main/elq
# but removed some of the code paths that I wasn't using.

def load_default_entity_linking_models(args):
    logging.info(">> Loading ELQ models")
    models = _load_models(args)
    logging.info(">> Intiailizing tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return models, tokenizer


def get_default_args(
    data_type,
    logging_dir,
    repo_path,
    eval_batch_size=8,
    threshold=-4.5,
    num_cands=10,
    print_every=10,
    num_batches=None,
    split_num=-1,
):
    models_path = f"{repo_path}models/"
    save_preds_dir = f"{logging_dir}data/elq_full_preds__{data_type}/"
    if split_num != -1:
        save_preds_dir += f"{split_num}/"

    if not os.path.exists(save_preds_dir):
        os.makedirs(save_preds_dir)

    config = {
        "interactive": False,
        "biencoder_model": models_path + "elq_wiki_large.bin",
        "biencoder_config": models_path + "elq_large_params.txt",
        "cand_token_ids_path": models_path + "entity_token_ids_128.t7",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "eval_batch_size": eval_batch_size,
        "output_path": logging_dir,
        "faiss_index": "hnsw",
        "index_path": models_path + "faiss_hnsw_index.pkl",
        "num_cand_mentions": num_cands,
        "num_cand_entities": num_cands,
        "threshold_type": "joint",
        "threshold": threshold,
        "base_path": repo_path,
        "use_cuda": True,
        "save_preds_dir": save_preds_dir,
        "num_batches": num_batches,
        "print_every": print_every,
    }
    args = argparse.Namespace(**config)
    return args


# Taken and modified from BLINK/elq/main_dense.py#L97
def _load_candidates(
    entity_catalogue,
    entity_encoding,
    faiss_index="none",
    index_path=None,
    base_path="",
):
    # Only keep the things for hnsw
    assert faiss_index == "hnsw", "Unsupported index type!"

    candidate_encoding = None
    assert index_path is not None, "Error! Empty indexer path."
    indexer = DenseHNSWFlatIndexer(1)
    indexer.deserialize_from(index_path)

    candidate_encoding = torch.load(entity_encoding)
    if not os.path.exists(f"{base_path}models/id2title.json"):
        logging.info(">>  - candidate metadata doesn't exist, creating and dumping")
        id2title = {}
        id2text = {}
        id2wikidata = {}
        local_idx = 0
        with open(entity_catalogue, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                entity = json.loads(line)
                id2title[str(local_idx)] = entity["title"]
                id2text[str(local_idx)] = entity["text"]
                if "kb_idx" in entity:
                    id2wikidata[str(local_idx)] = entity["kb_idx"]
                local_idx += 1
        json.dump(id2title, open(f"{base_path}models/id2title.json", "w"))
        json.dump(id2text, open(f"{base_path}models/id2text.json", "w"))
        json.dump(id2wikidata, open(f"{base_path}models/id2wikidata.json", "w"))
    else:
        logging.info(">>  - loading id2title")
        id2title = json.load(open(f"{base_path}models/id2title.json"))
        logging.info(">>  - loading id2text")
        id2text = json.load(open(f"{base_path}models/id2text.json"))
        logging.info(">>  - loading id2wikidata")
        id2wikidata = json.load(open(f"{base_path}models/id2wikidata.json"))

    return (
        candidate_encoding,
        indexer,
        id2title,
        id2text,
        id2wikidata,
    )


# Taken from BLINK/elq/main_dense.py#L610
def _load_models(args):
    # Load biencoder model
    try:
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(args.biencoder_config) as json_file:
            for line in json_file:
                line = line.replace("'", '"')
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                biencoder_params = json.loads(line)
                break
    biencoder_params["path_to_model"] = args.biencoder_model
    biencoder_params["cand_token_ids_path"] = args.cand_token_ids_path
    biencoder_params["eval_batch_size"] = getattr(args, "eval_batch_size", 8)
    biencoder_params["no_cuda"] = (
        not getattr(args, "use_cuda", False) or not torch.cuda.is_available()
    )
    if biencoder_params["no_cuda"]:
        biencoder_params["data_parallel"] = False
    biencoder_params["load_cand_enc_only"] = False
    if getattr(args, "max_context_length", None) is not None:
        biencoder_params["max_context_length"] = args.max_context_length
    biencoder = load_biencoder(biencoder_params)
    if biencoder_params["no_cuda"] and type(biencoder.model).__name__ == "DataParallel":
        biencoder.model = biencoder.model.module
    elif (
        not biencoder_params["no_cuda"]
        and type(biencoder.model).__name__ != "DataParallel"
    ):
        biencoder.model = torch.nn.DataParallel(biencoder.model)
    model_device = next(biencoder.model.parameters()).device
    logging.info(
        f">> Biencoder model type: {type(biencoder.model)}, device: {model_device}"
    )
    logging.info(f">> Biencoder params: {biencoder_params}")

    # load candidate entities
    logging.info(">> Loading candidate entities")
    (candidate_encoding, indexer, id2title, id2text, id2wikidata,) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        args.faiss_index,
        args.index_path,
        base_path=args.base_path,
    )
    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        indexer,
        id2title,
        id2text,
        id2wikidata,
    )


# Taken from BLINK/elq/main_dense.py#L193
def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    """
    Samples: list of examples, each of the form--
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    period_token = 1012
    max_sample_len = biencoder_params["max_context_length"] - 2 # [101] + sample + [102]
    samples_text_tuple = []
    all_start_inds = []
    all_end_inds = []
    title_lens = []
    content_lens = []

    max_seq_len = 0
    for sample in samples:
        if 'content' in sample:
            # Wiki sample: "id" "title" "content"
            # Q sample: "id" "text"
            title_encoded = tokenizer.encode_plus(
                sample['title'],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            content_encoded = tokenizer.encode_plus(
                sample["content"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            all_tokens = title_encoded['input_ids'] + [period_token] + content_encoded['input_ids']
            title_lens.append(len(title_encoded['input_ids']))
            content_lens.append(len(content_encoded['input_ids']))

            title_max_offset = title_encoded['offset_mapping'][-1][-1]
            period_start = title_max_offset
            period_end = title_max_offset + 1
            start_content_offset = period_end + 1 # for space
            all_offsets = (
                title_encoded['offset_mapping'] +
                [(period_start, period_end)] + 
                [(
                    s + start_content_offset, e + start_content_offset
                ) for s, e in content_encoded['offset_mapping']]
            )

            if len(all_tokens) > max_sample_len and False:
                print(">> error, input too long")
            truncated_ids = all_tokens[:max_sample_len]
            start_inds = [o[0] for o in all_offsets[:max_sample_len]]
            end_inds = [o[1] for o in all_offsets[:max_sample_len]]

            # truncate the end if the sequence is too long...
            encoded_sample = [101] + truncated_ids + [102]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            padding = [
                0
                for _ in range(
                biencoder_params["max_context_length"] - len(encoded_sample)
                )
            ]

            all_start_inds.append(start_inds + [*padding])
            all_end_inds.append(end_inds + [*padding])
        else:
            all_start_inds = None
            all_end_inds = None
            ids = tokenizer.encode(sample['text'])
            truncated_ids = ids[:max_sample_len]
            encoded_sample = [101] + truncated_ids + [102]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            padding = [
                0
                for _ in range(
                    biencoder_params["max_context_length"] - len(encoded_sample)
                )
            ]

        samples_text_tuple.append(encoded_sample + [*padding])

    tensor_data_tuple = [torch.tensor(samples_text_tuple)]
    if all_start_inds is not None:
        tensor_data_tuple.extend([
            torch.tensor(all_start_inds),
            torch.tensor(all_end_inds),
            torch.tensor(title_lens),
            torch.tensor(content_lens),
        ])

    tensor_data = TensorDataset(*tensor_data_tuple)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _save_biencoder_outs(
    save_preds_dir,
    nns,
    dists,
    pred_mention_bounds,
    cand_scores,
    mention_scores,
    runtime,
):
    if not os.path.exists(save_preds_dir):
        os.makedirs(save_preds_dir, exist_ok=True)
    np.save(os.path.join(save_preds_dir, "biencoder_nns.npy"), nns)
    np.save(os.path.join(save_preds_dir, "biencoder_dists.npy"), dists)
    np.save(
        os.path.join(save_preds_dir, "biencoder_mention_bounds.npy"),
        pred_mention_bounds,
    )
    np.save(os.path.join(save_preds_dir, "biencoder_cand_scores.npy"), cand_scores)
    np.save(
        os.path.join(save_preds_dir, "biencoder_mention_scores.npy"), mention_scores
    )
    with open(os.path.join(save_preds_dir, "runtime.txt"), "w") as wf:
        wf.write(str(runtime))


def _load_biencoder_outs(save_preds_dir):
    nns = np.load(os.path.join(save_preds_dir, "biencoder_nns.npy"), allow_pickle=True)
    dists = np.load(
        os.path.join(save_preds_dir, "biencoder_dists.npy"), allow_pickle=True
    )
    pred_mention_bounds = np.load(
        os.path.join(save_preds_dir, "biencoder_mention_bounds.npy"), allow_pickle=True
    )
    cand_scores = np.load(
        os.path.join(save_preds_dir, "biencoder_cand_scores.npy"), allow_pickle=True
    )
    mention_scores = np.load(
        os.path.join(save_preds_dir, "biencoder_mention_scores.npy"), allow_pickle=True
    )
    runtime = float(open(os.path.join(save_preds_dir, "runtime.txt")).read())
    return nns, dists, pred_mention_bounds, cand_scores, mention_scores, runtime


def _run_biencoder(
    args,
    biencoder,
    dataloader,
    candidate_encoding,
    samples,
    num_cand_mentions=50,
    num_cand_entities=10,
    device="cpu",
    sample_to_all_context_inputs=None,
    threshold=0.0,
    indexer=None,
):
    """
    Returns: tuple
        labels (List[int])
            [(max_num_mentions_gold) x exs]:
            gold labels -- returns None if no labels

        nns (List[Array[int]])
            [(# of pred mentions, cands_per_mention) x exs]:
            predicted entity IDs in each example

        dists (List[Array[float]])
            [(# of pred mentions, cands_per_mention) x exs]:
            scores of each entity in nns

        pred_mention_bounds (List[Array[int]])
            [(# of pred mentions, 2) x exs]:
            predicted mention boundaries in each examples

        mention_scores (List[Array[float]])
            [(# of pred mentions,) x exs]:
            mention score logit

        cand_scores (List[Array[float]])
            [(# of pred mentions, cands_per_mention) x exs]:
            candidate score logit
    """
    biencoder.model.eval()
    # biencoder_model = biencoder.model
    # if hasattr(biencoder.model, "module"):
    #    biencoder_model = biencoder.model.module

    context_inputs = []
    nns = []
    dists = []
    # mention_dists = []
    pred_mention_bounds = []
    mention_scores = []
    cand_scores = []
    # sample_idx = 0
    # ctxt_idx = 0
    # label_ids = None
    for step, batch in enumerate(dataloader):
        if step % args.print_every == 0:
            perc_complete = -1.0
            if hasattr(args, "num_batches"):
                perc_complete = step * 100.0 / args.num_batches
            logging.info(
                f">> [{perc_complete:0.2f}%] Processing {step} "
                + f"out of {args.num_batches} batches"
            )

        context_input = batch[0].to(device)
        mask_ctxt = context_input != biencoder.NULL_IDX
        with torch.no_grad():
            context_outs = biencoder.encode_context(
                context_input,
                num_cand_mentions=num_cand_mentions,
                topK_threshold=threshold,
            )
            embedding_ctxt = context_outs["mention_reps"]
            left_align_mask = context_outs["mention_masks"]
            chosen_mention_logits = context_outs["mention_logits"]
            chosen_mention_bounds = context_outs["mention_bounds"]

            """
            GET TOP CANDIDATES PER MENTION
            """
            # (all_pred_mentions_batch, embed_dim)
            embedding_ctxt = embedding_ctxt[left_align_mask]
            if indexer is None:
                logging.info(">> This doesn't happen right??")
                try:
                    cand_logits, _, _ = biencoder.score_candidate(
                        context_input,
                        None,
                        text_encs=embedding_ctxt,
                        cand_encs=candidate_encoding.to(device),
                    )
                    # DIM (all_pred_mentions_batch, num_cand_entities);
                    #     (all_pred_mentions_batch, num_cand_entities)
                    top_cand_logits_shape, top_cand_indices_shape = cand_logits.topk(
                        num_cand_entities, dim=-1, sorted=True
                    )
                except:  # noqa: E722
                    # for memory savings, go through one chunk of candidates at a time
                    SPLIT_SIZE = 1000000
                    done = False
                    while not done:
                        top_cand_logits_list = []
                        top_cand_indices_list = []
                        max_chunk = int(len(candidate_encoding) / SPLIT_SIZE)
                        for chunk_idx in range(max_chunk):
                            try:
                                # DIM (num_total_mentions, num_cand_entities);
                                #     (num_total_mention, num_cand_entities)
                                top_cand_logits, top_cand_indices = embedding_ctxt.mm(
                                    candidate_encoding[
                                        chunk_idx
                                        * SPLIT_SIZE : (chunk_idx + 1)
                                        * SPLIT_SIZE
                                    ]
                                    .to(device)
                                    .t()
                                    .contiguous()
                                ).topk(10, dim=-1, sorted=True)
                                top_cand_logits_list.append(top_cand_logits)
                                top_cand_indices_list.append(
                                    top_cand_indices + chunk_idx * SPLIT_SIZE
                                )
                                if (
                                    len(
                                        (top_cand_indices_list[chunk_idx] < 0).nonzero()
                                    )
                                    > 0
                                ):
                                    import pdb

                                    pdb.set_trace()
                            except:  # noqa: E722
                                SPLIT_SIZE = int(SPLIT_SIZE / 2)
                                break
                        if len(top_cand_indices_list) == max_chunk:
                            # DIM (num_total_mentions, num_cand_entities);
                            #     (num_total_mentions, num_cand_entities) -->
                            #       top_top_cand_indices_shape indexes
                            #       into top_cand_indices
                            (
                                top_cand_logits_shape,
                                top_top_cand_indices_shape,
                            ) = torch.cat(top_cand_logits_list, dim=-1).topk(
                                num_cand_entities, dim=-1, sorted=True
                            )
                            # make indices index into candidate_encoding
                            # DIM (num_total_mentions, max_chunk*num_cand_entities)
                            all_top_cand_indices = torch.cat(
                                top_cand_indices_list, dim=-1
                            )
                            # DIM (num_total_mentions, num_cand_entities)
                            top_cand_indices_shape = all_top_cand_indices.gather(
                                -1, top_top_cand_indices_shape
                            )
                            done = True
            else:
                # DIM (all_pred_mentions_batch, num_cand_entities);
                #     (all_pred_mentions_batch, num_cand_entities)
                top_cand_logits_shape, top_cand_indices_shape = indexer.search_knn(
                    embedding_ctxt.cpu().numpy(), num_cand_entities
                )
                top_cand_logits_shape = torch.tensor(top_cand_logits_shape).to(
                    embedding_ctxt.device
                )
                top_cand_indices_shape = torch.tensor(top_cand_indices_shape).to(
                    embedding_ctxt.device
                )

            # DIM (bs, max_num_pred_mentions, num_cand_entities)
            top_cand_logits = torch.zeros(
                chosen_mention_logits.size(0),
                chosen_mention_logits.size(1),
                top_cand_logits_shape.size(-1),
            ).to(top_cand_logits_shape.device, top_cand_logits_shape.dtype)
            top_cand_logits[left_align_mask] = top_cand_logits_shape
            top_cand_indices = torch.zeros(
                chosen_mention_logits.size(0),
                chosen_mention_logits.size(1),
                top_cand_indices_shape.size(-1),
            ).to(top_cand_indices_shape.device, top_cand_indices_shape.dtype)
            top_cand_indices[left_align_mask] = top_cand_indices_shape

            """
            COMPUTE FINAL SCORES FOR EACH CAND-MENTION PAIR + PRUNE USING IT
            """
            # Has NAN for impossible mentions...
            # log p(entity && mb) = log [p(entity|mention bounds) *
            #             p(mention bounds)] = log p(e|mb) + log p(mb)
            # DIM (bs, max_num_pred_mentions, num_cand_entities)
            scores = (
                torch.log_softmax(top_cand_logits, -1)
                + torch.sigmoid(chosen_mention_logits.unsqueeze(-1)).log()
            )

            """
            DON'T NEED TO RESORT BY NEW SCORE -- DISTANCE
            PRESERVING (largest entity score still be largest entity score)
            """

            for idx in range(len(batch[0])):
                # [(seqlen) x exs] <= (bsz, seqlen)
                context_inputs.append(
                    context_input[idx][mask_ctxt[idx]].data.cpu().numpy()
                )
                # [(max_num_mentions, cands_per_mention) x exs] <=
                #       (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                nns.append(
                    top_cand_indices[idx][left_align_mask[idx]].data.cpu().numpy()
                )
                # [(max_num_mentions, cands_per_mention) x exs] <=
                #       (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                dists.append(scores[idx][left_align_mask[idx]].data.cpu().numpy())
                # [(max_num_mentions, 2) x exs] <=
                #      (bsz, max_num_mentions=num_cand_mentions, 2)
                pred_mention_bounds.append(
                    chosen_mention_bounds[idx][left_align_mask[idx]].data.cpu().numpy()
                )
                # [(max_num_mentions,) x exs] <=
                #      (bsz, max_num_mentions=num_cand_mentions)
                mention_scores.append(
                    chosen_mention_logits[idx][left_align_mask[idx]].data.cpu().numpy()
                )
                # [(max_num_mentions, cands_per_mention) x exs] <=
                #      (bsz, max_num_mentions=num_cand_mentions, cands_per_mention)
                cand_scores.append(
                    top_cand_logits[idx][left_align_mask[idx]].data.cpu().numpy()
                )

    return (
        nns, 
        dists, 
        pred_mention_bounds, 
        mention_scores, 
        cand_scores
    )



# Taken from BLINK/elq/main_dense.py#L380
def _get_and_save_predictions(
    args,
    tokenizer,
    dataloader,
    biencoder_params,
    samples,
    nns,
    dists,
    mention_scores,
    cand_scores,
    pred_mention_bounds,
    id2title,
    threshold=-2.9,
    mention_threshold=-0.6931,
):
    """
    Arguments:
        args, dataloader, biencoder_params, samples, nns, dists, pred_mention_bounds
    Returns:
        all_entity_preds,
        num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window,
            num_correct_strong_from_input_window, num_gold_from_input_window
    """

    # save biencoder predictions and print precision/recalls
    num_correct_weak = 0
    num_correct_strong = 0
    num_predicted = 0
    num_gold = 0
    num_correct_weak_from_input_window = 0
    num_correct_strong_from_input_window = 0
    num_gold_from_input_window = 0
    all_entity_preds = []

    save_biencoder_file = os.path.join(args.save_preds_dir, "biencoder_outs.jsonl")
    out_file = open(save_biencoder_file, "a+")
    logging.info(f">> Start saving results to: {save_biencoder_file}")

    # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
    # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
    # cand_scores (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # mention_scores (List[Array[float]]) [(num_pred_mentions,) x exs])
    for batch_num, batch_data in enumerate(dataloader):
        batch_context = batch_data[0]
        if len(batch_data) > 1:
            (
                _,
                start_inds,
                end_inds,
                title_lens,
                content_lens,
            ) = batch_data
        for b in range(len(batch_context)):
            i = batch_num * biencoder_params["eval_batch_size"] + b
            sample = samples[i]
            input_context = batch_context[b][
                batch_context[b] != 0
            ].tolist()  # filter out padding
            if len(batch_data) > 1:
                # "Title. Content"
                start_offset = 1 # 101
                title_len = title_lens[b].item()
                sep_len = 1
                content_len = content_lens[b].item()
                b_start_inds = start_inds[b].tolist()
                b_end_inds = end_inds[b].tolist()
                title_s = start_offset
                title_e = start_offset + title_len
                content_s = title_e + sep_len
                content_e = len(input_context) - 1 # remove 102
                if False: # This asserts no truncation
                    assert content_e == content_s + content_len
                title_toks = input_context[title_s:title_e]
                content_toks = input_context[content_s:content_e]
                title_tok_offsets = [
                    (
                        b_start_inds[i], b_end_inds[i]
                    ) for i in range(title_s-start_offset, title_e - start_offset)
                ]
                content_offset = len(sample['title'] + '. ')
                content_tok_offsets = [
                    (b_start_inds[i] - content_offset, b_end_inds[i] - content_offset) for i in range(
                        content_s - start_offset, content_e - start_offset
                    )
                ]
                #title_strs = [sample['title'][s:e] for s, e in title_tok_offsets]
                #content_strs = [sample['content'][s:e] for s, e in content_tok_offsets]

            # (num_pred_mentions, cands_per_mention)
            scores = dists[i] if args.threshold_type == "joint" else cand_scores[i]
            cands_mask = scores[:, 0] == scores[:, 0]
            pred_entity_list = nns[i][cands_mask]
            # if len(pred_entity_list) > 0:
            #    e_id = pred_entity_list[0]
            distances = scores[cands_mask]
            # (num_pred_mentions, 2)
            entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]

            if args.threshold_type == "joint":
                # THRESHOLDING
                top_mentions_mask = distances[:, 0] > threshold
            elif args.threshold_type == "top_entity_by_mention":
                top_mentions_mask = mention_scores[i] > mention_threshold
            elif args.threshold_type == "thresholded_entity_by_mention":
                top_mentions_mask = (distances[:, 0] > threshold) & (
                    mention_scores[i] > mention_threshold
                )

            _, sort_idxs = torch.tensor(distances[:, 0][top_mentions_mask]).sort(
                descending=True
            )
            # cands already sorted by score
            all_pred_entities = pred_entity_list[:, 0][top_mentions_mask]
            e_mention_bounds = entity_mention_bounds_idx[top_mentions_mask]
            chosen_distances = distances[:, 0][top_mentions_mask]
            if len(all_pred_entities) >= 2:
                all_pred_entities = all_pred_entities[sort_idxs]
                e_mention_bounds = e_mention_bounds[sort_idxs]
                chosen_distances = chosen_distances[sort_idxs]

            # prune mention overlaps
            e_mention_bounds_pruned = []
            all_pred_entities_pruned = []
            chosen_distances_pruned = []
            mention_masked_utterance = np.zeros(len(input_context))
            # ensure well-formed-ness, prune overlaps
            # greedily pick highest scoring, then prune all overlapping
            for idx, mb in enumerate(e_mention_bounds):
                mb[1] += 1  # prediction was inclusive, now make exclusive
                # check if in existing mentions
                if (
                    args.threshold_type != "top_entity_by_mention"
                    and mention_masked_utterance[mb[0] : mb[1]].sum() >= 1
                ):
                    continue
                e_mention_bounds_pruned.append(mb)
                all_pred_entities_pruned.append(all_pred_entities[idx])
                chosen_distances_pruned.append(float(chosen_distances[idx]))
                mention_masked_utterance[mb[0] : mb[1]] = 1

            input_context = input_context[1:-1]  # remove BOS and sep
            pred_triples = [
                (
                    str(all_pred_entities_pruned[j]),
                    int(e_mention_bounds_pruned[j][0]) - 1,  # -1 for BOS
                    int(e_mention_bounds_pruned[j][1]) - 1,
                )
                for j in range(len(all_pred_entities_pruned))
            ]

            entity_results = {
                "id": sample["id"],
                "scores": chosen_distances_pruned,
            }
            if 'text' in sample:
                entity_results.update({
                    "text": sample['text'],
                    "pred_tuples_string": [
                        [
                            id2title[triple[0]],
                            tokenizer.decode(input_context[triple[1] : triple[2]]),
                        ]
                        for triple in pred_triples
                    ],
                    "pred_triples": pred_triples,
                    "tokens": input_context,
                })
            else:
                title_pred_triples = []
                content_pred_triples = []
                title_pred_tuples_string = []
                content_pred_tuples_string = []
                # title_len, content_len
                for triple in pred_triples:
                    ent_id, tok_s, tok_e = triple
                    if tok_e <= title_len + start_offset:
                        title_tok_s = tok_s
                        title_tok_e = tok_e
                        title_str_s = title_tok_offsets[title_tok_s][0]
                        title_str_e = title_tok_offsets[title_tok_e - 1][1]
                        title_pred_triples.append((
                            ent_id, title_tok_s, title_tok_e
                        ))
                        title_pred_tuples_string.append([
                            id2title[ent_id],
                            sample['title'][title_str_s:title_str_e]
                        ])
                    else:
                        cont_tok_s = tok_s - title_len - sep_len
                        cont_tok_e = tok_e - title_len - sep_len
                        cont_str_s = content_tok_offsets[cont_tok_s][0]
                        cont_str_e = content_tok_offsets[cont_tok_e - 1][1]
                        content_pred_triples.append((
                            ent_id, cont_tok_s, cont_tok_e
                        ))
                        content_pred_tuples_string.append([
                            id2title[ent_id],
                            sample['content'][cont_str_s:cont_str_e]
                        ])
                    
                entity_results.update({
                    'title': sample['title'],
                    'content': sample['content'],
                    'title_toks': title_toks,
                    'content_toks': content_toks,
                    'title_tok_text_offsets': title_tok_offsets,
                    'content_tok_text_offsets': content_tok_offsets,
                    'title_pred_triples': title_pred_triples,
                    'title_pred_tuples_string': title_pred_tuples_string,
                    'content_pred_triples': content_pred_triples,
                    'content_pred_tuples_string': content_pred_tuples_string,
                })

            all_entity_preds.append(entity_results)
            out_file.write(json.dumps(entity_results) + "\n")

    out_file.close()


def run_elq(
    args,
    tokenizer,
    biencoder,
    biencoder_params,
    candidate_encoding,
    indexer,
    id2title,
    id2text,
    id2wikidata,
    samples,
):
    assert args.save_preds_dir, "Need to specify args.save_preds_dir"
    # stopping_condition = False
    threshold = float(args.threshold)
    if args.threshold_type == "top_entity_by_mention":
        assert args.mention_threshold is not None
        mention_threshold = float(args.mention_threshold)
    else:
        mention_threshold = threshold

    logging.info(">> Preparing data for biencoder")
    dataloader = _process_biencoder_dataloader(
        samples,
        tokenizer,
        biencoder_params,
    )

    assert getattr(args, "save_preds_dir", None) is not None
    logging.info(">> Running biencoder")
    start_time = time.time()
    (
        nns, dists, pred_mention_bounds, mention_scores, cand_scores
    ) = _run_biencoder(
        args,
        biencoder,
        dataloader,
        candidate_encoding,
        samples=samples,
        num_cand_mentions=args.num_cand_mentions,
        num_cand_entities=args.num_cand_entities,
        device="cpu" if biencoder_params["no_cuda"] else "cuda",
        threshold=mention_threshold,
        indexer=indexer,
    )
    end_time = time.time()
    logging.info(">> Finished running biencoder")

    runtime = end_time - start_time

    assert (
        len(samples)
        == len(nns)
        == len(dists)
        == len(pred_mention_bounds)
        == len(cand_scores)
        == len(mention_scores)
    )

    # Part 2: get predictions from biencoder output
    _get_and_save_predictions(
        args,
        tokenizer,
        dataloader,
        biencoder_params,
        samples,
        nns,
        dists,
        mention_scores,
        cand_scores,
        pred_mention_bounds,
        id2title,
        threshold=threshold,
        mention_threshold=mention_threshold,
    )


