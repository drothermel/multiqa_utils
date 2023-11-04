import copy
import logging
import json

from pyserini.search.lucene import LuceneSearcher

import utils.file_utils as fu
import utils.run_utils as ru
import multiqa_utils.data_utils as du


# Deterministic ordering to allow for sharding
def data_to_query_list(data_list, id_fxn, text_fxn):
    query_list = [
        {
            "id": id_fxn(v),
            "text": text_fxn(v),
            "data": v,
        }
        for v in data_list
    ]
    query_list = sorted(
        query_list,
        key=lambda d: d["id"],
    )
    return proof_query_list


def searcher_out_to_list_of_dicts(searcher_out):
    return [
        {
            "score": so.score,
            **json.loads(so.raw),
        }
        for so in searcher_out
    ]


def bm25_batch_query_dump(
    index_path,
    query_data_list,
    num_threads,
    top_k,
    batch_size,
    output_base,
    id_key="id",
    query_key="text",
):
    logging.info(f">> Loading index: {index_path}")
    searcher = LuceneSearcher(index_path)

    # Batching to avoid OOMs (list of lists)
    batched_queries = ru.batch_list(query_data_list, batch_size)
    num_batches = len(query_data_list) // batch_size + 1

    logging.info(">> Running query")
    for i, query_batch in enumerate(batched_queries):
        ru.processed_log(i, num_batches)

        # process input into query_list and qid_list and verify max len
        query_list = [qd[query_key] for qd in query_batch]
        query_list = [q if len(q) < 5000 else q[-5000:] for q in query_list]
        qid_list = [qd[id_key] for qd in query_batch]
        searcher_out = searcher.batch_search(
            queries=query_list,
            qids=qid_list,
            threads=num_threads,
            k=top_k,
        )

        batch_query_with_results = []
        for qd in query_batch:
            query_id = qd[id_key]
            query_out_list = searcher_out_to_list_of_dicts(searcher_out[query_id])
            query_out_w_data = copy.deepcopy(qd)
            query_out_w_data["ctxs"] = query_out_list
            batch_query_with_results.append(query_out_w_data)
        # Dump for each batch
        fu.dumpjsonl(
            batch_query_with_results,
            f"{output_base}_{i}.jsonl",
            verbose=False,
        )


if __name__ == "__main__":
    print("Loading")
    import hydra as hyd
    hyd.initialize(version_base=None, config_path="../scripts/conf")
    cfg = hyd.compose(config_name="maqa")
    qmp_dev = du.get_data(cfg, "qmp_dev")

    print("Preprocessing")
    test_qd = [qd for i, qd in enumerate(qmp_dev) if i < 10]
    proof_data = du.qmp_raw_to_proof_info(test_qd)
    proof_query_list = data_to_query_list(proof_data)
    index_path = "/scratch/ddr8143/wikipedia/indexes/qampari_wikipedia_chunked_fixed_v0"

    print("Running")
    proofs_with_context = bm25_batch_query_dump(
        index_path,
        proof_query_list,
        10,
        1,
        1000,
        './tmp',
        id_key="pid",
        query_key="proof",
    )
    breakpoint()
