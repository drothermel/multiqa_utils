import jsonlines

import elq.main_dense as main_dense


def load_default_entity_linking_models(
    output_path,
    repo_path="/scratch/ddr8143/repos/BLINK/"
):
    models_path = f"{repo_path}models/"
    config = {
        "interactive": False,
        "biencoder_model": models_path+"elq_wiki_large.bin",
        "biencoder_config": models_path+"elq_large_params.txt",
        "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "output_path": output_path, # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path+"faiss_hnsw_index.pkl",
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


def elq_tag_data_and_dump(
    config_namespace, models, dlist, outfile,
    id_key='qid', text_key='question_text', chunk_size=10,
):
    assert not os.path.exists(outfile)
        
    new_data_to_link = []
    with jsonlines.open(outfile, mode='w') as writer:
        for i in range(len(dlist)):
            if len(new_data_to_link) < chunk_size:
                new_data_to_link.append({"id": dlist[i][id_key], "text": dlist[i][text_key]})
            else:
                preds = main_dense.run(config_namespace, None, *models, test_data=new_data_to_link)
                for p in preds:
                    writer.write(p)
                new_data_to_link = []
        
        # Score and dump the final set
        preds = main_dense.run(args, None, *models, test_data=new_data_to_link)
        for p in preds:
            writer.write(p)
    print(">> Wrote all entity links to:", outfile)