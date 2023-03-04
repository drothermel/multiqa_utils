import json
import jsonlines
import os
import time
from collections import defaultdict

import multiqa_utils.openai_utils as ou
import multiqa_utils.qampari_utils as qu
import multiqa_utils.general_utils as gu

## =============================================== ##
## ==================== Loading ================== ##
## =============================================== ##


def load_manual_train_decomp(dataset):
    if dataset == "qampari":
        dpath = qu.MANUAL_TRAIN_DECOMPOSITION_PATH
    else:
        assert False, f"No decompositions for dataset: {dataset}"

    str_keys = json.load(open(dpath))
    int_keys = {int(k): v for k, v in str_keys.items()}
    return int_keys


## =============================================== ##
## ==================== Reading ================== ##
## =============================================== ##


def get_gt_type_from_qid(qid):
    if "simple" in qid:
        return "simple"
    elif "comp" in qid:
        return "composition"
    else:
        return "intersection"


def get_pred_type_from_decomp(decomp_lines):
    if "[ANS1]" in decomp_lines[-1]:
        return "composition"
    elif "None" in decomp_lines[-1]:
        return "simple"
    return "intersection"


def prompt_output_to_eval_dict(init_dict, qprompt, result_text, manual_check=False):
    prompt_lines = qprompt.split("\n")
    ans_lines = result_text.split("\n")

    gt_type = get_gt_type_from_qid(init_dict["qid"])
    pred_type = get_pred_type_from_decomp(ans_lines)
    correct_type = gt_type == pred_type

    final_output = {
        **init_dict,
        "q_ind": int(init_dict["qid"].split("_")[0]),
        "decomp_prompt": qprompt,
        "decomp_res_text": result_text,
        "gt_type": gt_type,
        "pred_type": pred_type,
        "pred_type_correct": correct_type,
    }

    if manual_check:
        note = ""
        decomp_correct = False
        if gt_type == "simple" and correct_type:
            print(">> Simple question, type correct, skip manual check")
        else:
            if correct_type:
                print(">> Pred type correct:", pred_type)
            else:
                print(f">> Pred type wrong: {pred_type} when gt was {gt_type}")

            print()
            print(prompt_lines[-2])
            print(prompt_lines[-1] + ans_lines[0])
            for l in ans_lines[1:]:
                print(l)
            print()

            if correct_type:
                decomp_correct = (
                    input("Is the decomposition correct? ('T' for yes) >>") == "T"
                )
            note = input("Any notes on this example? >>")
            print("----------------------\n")
        final_output["note"] = note
        final_output["decomp_correct"] = decomp_correct
    return final_output


## =============================================== ##
## =============== Compound Commands ============= ##
## =============================================== ##


def decompose_indmap_with_prompt(
    to_eval_dev_inds, 
    outfile,
    dataset="qampari", progress_increment=10,
    engine='text-davinci-003',
):
    assert dataset == "qampari"
    manual_train_decomp = load_manual_train_decomp(dataset)
    qmp_train = qu.load_wikidata_train_data()
    qmp_dev = qu.load_wikidata_dev_data()

    if os.path.exists(outfile):
        print("Won't overwrite:", outfile)
        return

    with jsonlines.open(outfile, mode="w") as writer:
        for qtype, inds in to_eval_dev_inds.items():
            print(f"Starting to decompose {qtype}, {len(inds)} queries to go")
            for i, qid in enumerate(inds):
                if i % progress_increment == 0:
                    print(f">> elem {i}")

                qdata = qmp_dev[qid]
                qprompt = get_qmp_decomp_prompt_v1(
                    qmp_train, manual_train_decomp, qmp_dev[qid]
                )
                _, res_text = ou.prompt_openai(qprompt, engine=engine)
                q_output = prompt_output_to_eval_dict(
                    qdata,
                    qprompt,
                    res_text,
                )
                writer.write(q_output)
    print("Fnished Decomp & Wrote:", outfile)
    

    
def decompose_with_prompt(
    data_to_decompose, 
    outfile,
    dataset="qampari",
    progress_increment=10,
    engine='text-davinci-003',
    rate_limit=19,
):
    assert dataset == "qampari"
    manual_train_decomp = load_manual_train_decomp(dataset)
    qmp_train = qu.load_wikidata_train_data()
    mode = 'w'
    
    if os.path.exists(outfile):
        mode = 'a'
        already_decomp = set([d['qid'] for d in gu.loadjsonl(outfile)])
        print("Initial data len:", len(data_to_decompose))
        data_to_decompose = [d for d in data_to_decompose if d['qid'] not in already_decomp]
        print("  - after loading:", len(already_decomp), "new len:", len(data_to_decompose))

    
    time_per_query = 60.0 / rate_limit
    total_start = time.time()
    start = time.time()
    with jsonlines.open(outfile, mode=mode) as writer:
        for i, qdata in enumerate(data_to_decompose):
            if i % progress_increment == 0:
                print(f">> [{(time.time() - total_start)/60.0:0.1f}] elem {i:,} / {len(data_to_decompose):,}")
            
            qprompt = get_qmp_decomp_prompt_v1(
                qmp_train, manual_train_decomp, qdata
            )
            _, res_text = ou.prompt_openai(qprompt, engine=engine)
            while res_text is None:
                print("  -> Hit strange rate limit, sleep for two minutes and try again.")
                time.sleep(120)
                _, res_text = ou.prompt_openai(qprompt, engine=engine)
                print("        ===> okkk, lets go!")
            
            q_output = prompt_output_to_eval_dict(
                qdata,
                qprompt,
                res_text,
            )
            writer.write(q_output)
            end = time.time()
            print("Raw time:", end - start)
            while end - start < time_per_query:
                time.sleep(5.0)
                end = time.time()
                #print("  - New time:", end - start)
            print("  -> Total time:", end-start)
            start = end
    print("Fnished Decomp & Wrote:", outfile)


def process_prompt_outputs(
    query_output_file,
    manual_check=False,
):
    reader = jsonlines.Reader(open(query_output_file))
    incorrect_ids = defaultdict(list)
    num_qs = defaultdict(int)
    num_correct_type = defaultdict(int)
    num_correct_decomp = defaultdict(int)
    failed_queries = []

    new_outputs = []
    for dd in reader:
        if dd["decomp_res_text"] is None:
            failed_queries.append(dd)
            continue

        nout = prompt_output_to_eval_dict(
            init_dict=dd,
            qprompt=dd["decomp_prompt"],
            result_text=dd["decomp_res_text"],
            manual_check=manual_check,
        )
        num_qs[nout["gt_type"]] += 1
        if nout["pred_type_correct"]:
            num_correct_type[nout["gt_type"]] += 1
        else:
            incorrect_ids[nout["gt_type"]].append((dd["q_ind"], nout["pred_type"]))

        if manual_check:
            new_outputs.append(nout)
            if nout["decomp_correct"]:
                num_correct_decomp[nout["gt_type"]] += 1

    stats = {
        "failed_queries": failed_queries,
        "incorrect_ids": dict(incorrect_ids),
        "num_total": dict(num_qs),
        "num_correct_type": dict(num_correct_type),
        "correct_type_percent": {
            k: num_correct_type[k] * 100.0 / num_qs[k] for k in num_qs.keys()
        },
    }
    if manual_check:
        stats["num_correct_decomp"] = dict(num_correct_decomp)
        stats["correct_decomp_percent"] = {
            k: num_correct_decomp[k] * 100.0 / num_qs[k] for k in num_qs.keys()
        }

    return stats, new_outputs


## =============================================== ##
## ==================== Prompts ================== ##
## =============================================== ##


def qdata_to_print_prompt_v0(
    all_qdata, qdecompose, num_each=3, include_simple=False, shuffle=True
):
    prompt_list = []
    if include_simple:
        i = 0
        for qd in all_qdata:
            if i == num_each:
                break
            if "simple" in qd["qid"]:
                prompt_list.append(
                    """
Question: {init_q}
Can this be decomposed: No.""".format(
                        init_q=qd["question_text"],
                    )
                )
                i += 1

    comp_dec = {
        k: v for k, v in qdecompose.items() if v["question_type"] == "composition"
    }
    int_dec = {
        k: v for k, v in qdecompose.items() if v["question_type"] == "intersection"
    }

    for dec_set in [comp_dec, int_dec]:
        i = 0
        for qtid, decomp in dec_set.items():
            if i == num_each:
                break
            prompt_list.append(
                """
Question: {init_q}
Can this be decomposed: Yes.
Is this a composition or intersection question: {qtype}.
Question 1: {subqs1}
Question 2: {subqs2}
So the final answers are: {answer_list}.""".format(
                    init_q=all_qdata[qtid]["question_text"],
                    qtype=decomp["question_type"],
                    subqs1=decomp["subquestions"][0],
                    subqs2=decomp["subquestions"][1],
                    answer_list=", ".join(
                        list(
                            set(
                                [
                                    a["answer_text"]
                                    for a in all_qdata[qtid]["answer_list"]
                                ]
                            )
                        )[:5]
                    ),
                )
            )
            i += 1

    if shuffle:
        random.shuffle(prompt_list)
    for p in prompt_list:
        print(p)


def get_qmp_decomp_prompt_base_v1(train_qdata, decomps):
    prompt_str = """Instructions:
Choose the question type out of: composition or intersection.

Simple questions only require one piece of information. Example:

“Question: Which software, art, etc. has Don Broco as performer?
Question Type: simple.
Explanation: This is a simple question because we only need to know what Don Broco has performed in.”

Composition questions require getting one answer and then getting more information about that answer. Example:

“Question: What are the dates of death of persons that were a member of the political party Australian Labor Party (Anti-Communist)?
Question Type: composition.
Explanation: This is a complosition question because we need to know who were members of the political party and then we need to get additional information about each of them.”

Intersection questions require getting answers to two questions and then combining them. Example:

“Question: Which film has M. G. Ramachandran as a member of its cast and has J. Jayalalithaa as a member of its cast?
Question Type: intersection.
Explanation: This is an intersection question because we need to combine the answers to the first question, which film has M. G. Ramachandran as a member of its cast, with the second question, which film has J. Jayalalithaa as a member of its cast.”

----------

Examples:

"""

    for qid in [
        53823,
        36854,
        16043,
        53774,
        55950,
        40308,
        22726,
        28673,
        4484,
        53887,
        29563,
    ]:
        q_text = train_qdata[qid]["question_text"].strip()
        if q_text[-1] != "?":
            q_text += "?"

        qprompt = "Question: " + q_text + "\n"
        qtstr = train_qdata[qid]["qid"]
        if "wikidata_simple" in qtstr:
            qprompt += "Question Type: simple.\n"
            qprompt += "Question 1: " + q_text + "\n"
            qprompt += "Question 2: None\n"
        else:
            qdc = decomps[qid]
            qtype = "composition" if "comp" in qtstr else "intersection"
            qprompt += f"Question Type: {qtype}.\n"
            qprompt += "Question 1: " + qdc["subquestions"][0] + "\n"
            qprompt += "Question 2: " + qdc["subquestions"][1] + "\n"
        prompt_str += qprompt + "\n"
    return prompt_str


def get_qmp_decomp_prompt_v1(train_qdata, decomps, prompt_qdata):
    prompt_base = get_qmp_decomp_prompt_base_v1(train_qdata, decomps)
    q_text = prompt_qdata["question_text"].strip()
    if q_text[-1] != "?":
        q_text += "?"
    return prompt_base + "Question: " + q_text + "\n" + "Question Type:"
