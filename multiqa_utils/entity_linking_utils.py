import jsonlines
import argparse
import os
import json

import elq.main_dense as main_dense


def load_default_entity_linking_models(
    output_path, repo_path="/scratch/ddr8143/repos/BLINK/"
):
    models_path = f"{repo_path}models/"
    config = {
        "interactive": False,
        "biencoder_model": models_path + "elq_wiki_large.bin",
        "biencoder_config": models_path + "elq_large_params.txt",
        "cand_token_ids_path": models_path + "entity_token_ids_128.t7",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "output_path": output_path,  # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path + "faiss_hnsw_index.pkl",
        "num_cand_mentions": 10,
        "num_cand_entities": 10,
        "threshold_type": "joint",
        "threshold": -4.5,
        "base_path": repo_path,
    }

    config_namespace = argparse.Namespace(**config)
    print(">> Loading models, may take a few minutes.")
    models = main_dense.load_models(config_namespace, logger=None)

    return config_namespace, models


def loadjsonl(filename):
    all_lines = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            all_lines.append(obj)
    return all_lines


def loadpkl(filename):
    return pickle.load(open(filename, "rb"))


def load_file(path, ending=None):
    if ending == "json" or path[-len(".json") :] == ".json":
        return json.load(open(path))
    elif ending == "jsonl" or path[-len(".jsonl") :] == ".jsonl":
        return loadjsonl(path)
    elif ending == "pkl" or path[-len(".pkl") :] == ".pkl":
        return loadpkl(path)
    elif ending == "txt" or path[-len(".txt") :] == ".txt":
        return open(path).readlines()
    else:
        print(">> Path exists but can't load ending:", path)
        return None


def elq_tag_data_and_dump(
    config_namespace,
    models,
    dlist,
    outfile,
    id_key="qid",
    text_key="question_text",
    chunk_size=10,
):
    mode = 'w+'
    print(f"Entity link {len(dlist):,} items.")
    if os.path.exists(outfile):
        mode = 'a'
        already_written = load_file(outfile)
        aw_set = set([a['id'] for a in already_written])
        dlist = [d for d in dlist if d[id_key] not in aw_set]
        print(f">> File already exists to loaded: {len(already_written):,} and remaining to extract: {len(dlist):,}")
        
    if len(dlist) == 0:
        print(">> All data already entity linked.")
        return

    new_data_to_link = []
    j = 0
    with jsonlines.Writer(open(outfile, mode=mode), flush=True) as writer:
        for i in range(len(dlist)):
            if len(new_data_to_link) < chunk_size:
                new_data_to_link.append(
                    {"id": dlist[i][id_key], "text": dlist[i][text_key]}
                )
            else:
                print(f">> Calling main_dense.run() for {j}th time with {len(new_data_to_link)} items")
                j += 1
                preds = main_dense.run(
                    config_namespace, None, *models, test_data=new_data_to_link
                )
                print(f">> Finished {j-1}th run and writing data")
                for p in preds:
                    writer.write(p)
                new_data_to_link = []

        # Score and dump the final set
        preds = main_dense.run(
            config_namespace, None, *models, test_data=new_data_to_link
        )
        for p in preds:
            writer.write(p)
    print(">> Wrote all entity links to:", outfile)
