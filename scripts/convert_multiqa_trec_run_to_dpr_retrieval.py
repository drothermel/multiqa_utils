# Note, written based on https://github.com/castorini/pyserini/blob/master/pyserini/eval/convert_trec_run_to_dpr_retrieval_run.py

import argparse
import json
import os
from tqdm import tqdm

from pyserini.search import get_topics
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

def ambigqa_get_topics(topics_path):
    with open(topics_path, 'r') as f:
        all_data = json.load(f)

    topics = {}
    for d in all_data:
        answers = []
        for anns in d['annotations']:
            if anns['type'] == 'multipleQAs':
                for qap in anns['qaPairs']:
                    answers.extend(qap['answer'])
            else:
                answers.extend(anns['answer'])
        int_id = int(d['id'])
        topics[int_id] = {"answers": list(set(answers)), **d}
    return topics

def trec_to_json_contexts(trec_file, qa_data, searcher, tokenizer, out_file, store_raw=True):
    retrieval = {}
    with open(trec_file) as f_in:
        for line in tqdm(f_in.readlines()):
            question_id, _, doc_id, _, score, _ = line.strip().split()
            question_id = int(question_id)
            question = qa_data[question_id]['question']
            answers = qa_data[question_id]['answers']
            if answers[0] == '"':
                answers = answers[1:-1].replace('""', '"')
            ctx = json.loads(searcher.doc(doc_id).raw())['contents']
            if question_id not in retrieval:
                retrieval[question_id] = {'question': question, 'answers': answers, 'contexts': []}
            title, text = ctx.split('\n')
            answer_exist = has_answers(text, answers, tokenizer, False)
            if store_raw:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id,
                     'score': score,
                     'text': ctx,
                     'has_answer': answer_exist}
                )
            else:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id, 'score': score, 'has_answer': answer_exist}
                )

    json.dump(retrieval, open(out_file, 'w'), indent=4)

def json_contexts_to_dpr_data(context_file, out_file, drop_no_pos,  n_hard=1):
    context_data = json.load(open(context_file))
    dpr_data = []
    for _, d in context_data.items():
        dpr_d = {k: v for k, v in d.items() if k != "contexts"}
        dpr_d['positive_ctxs'] = [c for c in d['contexts'] if c['has_answer']]
        if drop_no_pos and len(dpr_d['positive_ctxs']) == 0:
            continue

        dpr_d['hard_negative_ctxs'] = []
        for c in d['contexts']:
            if not c['has_answer']:
                dpr_d['hard_negative_ctxs'].append(c)
                if len(dpr_d['hard_negative_ctxs']) == n_hard:
                    break
        dpr_data.append(dpr_d)
    json.dump(dpr_data, open(out_file, 'w'), indent=4)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an TREC run to DPR retrieval result json.')
    parser.add_argument('--topics', required=True, help='topic name')
    parser.add_argument('--index', required=True, help='Anserini Index that contains raw')
    parser.add_argument('--input', required=True, help='Input TREC run file.')
    parser.add_argument('--output', required=True, help='Output DPR Retrieval json file.')
    parser.add_argument('--drop-no-pos', action='store_true', help='Store raw text of passage')
    args = parser.parse_args()

    if args.topics.endswith('.json'):
        if 'ambigqa' in args.topics:
            qas = ambigqa_get_topics(args.topics)
    else:
        qas = get_topics(args.topics)

    if os.path.exists(args.index):
        searcher = LuceneSearcher(args.index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(args.index)
    if not searcher:
        exit()

    # In: qa data to annotate, trec file w docids of retrieved docs
    # Out: qa data with a context list, including whether answer in context
    intermediate_out = f"{args.output}__contexts"
    if os.path.exists(intermediate_out):
        print(">> Contexts file exists, skipping bm25 search:", intermediate_out)
    else:
        trec_to_json_contexts(
            trec_file=args.input,
            qa_data=qas,
            searcher=searcher,
            tokenizer=SimpleTokenizer(),
            out_file=intermediate_out,
        )
        print(">> Wrote context retrieval to: {intermediate_out}")

    # In: qa data with a context list, including whether answer in context
    # Out: DPR style data (list not dict) w/ positive_ctxs and hard_negative_ctxs
    if os.path.exists(args.output):
        # Check if it contains no pos, if so and flag is set, mv and redo
        if args.drop_no_pos:
            dropped_no_pos = True
            dataset = json.load(open(args.output))
            for d in dataset:
                if len(d['positive_ctxs']) == 0:
                    print(">> Found an exisiting version with empty positive contexts, move and redo")
                    os.rename(args.output, f"{args.output}__w_no_pos")
                    json_contexts_to_dpr_data(
                        context_file=intermediate_out,
                        out_file=args.output,
                        drop_no_pos=args.drop_no_pos,
                    )
                    break
        else:
            print(">> Found an exisiting version, skipping the reformatting")
    else:
        json_contexts_to_dpr_data(
            context_file=intermediate_out,
            out_file=args.output,
            drop_no_pos=args.drop_no_pos,
        )
        print(f">> Wrote final dataset to: {args.output}")

    

