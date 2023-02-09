from collections import defaultdict

import matplotlib.pyplot as plt

# Top Level Fxns
# - plot_nll_validation_from_dpr_retriever_useful_lines
# - ioa_to_rankPatK

# =========== Parse Training Logs =========== #

## DPR Reader Training

# Very hacky script that aggregates results from multiple GPUs (average across GPUs)
def parse_epoch_em_from_dpr_reader_training_logs(filename, n_gpus=4):
    results = {}
    with open(filename) as f:
        i = 0
        e = 0
        for l in f:
            curr_EM = 0
            if "EM" in l:
                curr_EM += float(l.split()[-1])
                i += 1

            if i == n_gpus:
                i = 0
                results[e] = curr_EM / n_gpus
                e += 1
    return results


## DPR Retriever Training

LINES_TO_EXTRACT = {
    "checkpoint_model": "Reading saved model from",
    "train_datasets": "train_datasets: ",
    "dev_datasets": "dev_datasets: ",
    "aggregated_data_size": "Aggregated data size: ",
    "cleaned_data_size": "Total cleaned data size: ",
    "shard_info": "samples_per_shard=",
    "epoch_losses": "Epoch: ",
    "nll_validation": "NLL Validation: ",
    "best_checkpoint": "New Best validation checkpoint ",
    "avg_rank_validation": "Av.rank validation: average rank ",
}

# Produces list of each type of line logged, according to key above
def parse_lines_from_dpr_retriever_training_logs(filename):
    log_lines = open(filename).readlines()
    useful_lines = defaultdict(list)
    for l in log_lines:
        for k, v in LINES_TO_EXTRACT.items():
            if v in l:
                ldata = l.split(v)[-1].strip()
                vldata = v + ldata
                if vldata not in useful_lines[k]:
                    useful_lines[k].append(vldata)
    return useful_lines


def plot_nll_validation_from_dpr_retriever_useful_lines(useful_lines):
    y = [
        float(l.split("loss = ")[-1].split(". ")[0])
        for l in useful_lines["nll_validation"]
    ]
    x = [i + 1 for i in range(len(y))]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, y)

    plt.xlabel("Epoch")
    # plt.ylim([0, 100])
    plt.xlim([0, 30])
    plt.ylabel("Validation Loss")
    plt.title("AmbigQA Finetuning of NQ DPR Retrieval Model: Validation Loss")
    # plt.legend()
    plt.show()


# ===== Evaluate Retrieval Performance against GT ====== #

# Easy viz of one question/answer datapoint
def viz_correct_answers_context_list(data_point):
    answer_llist = list(data_point["ans_mappings"].values())
    print("Question:", data_point["question"])
    print("Answers:", answer_llist)
    rs = correct_answers_context_list(
        context_list=data_point["ctxs"],
        answer_llist=answer_llist,
    )
    print("-------")
    returned_ans = rs["correct_answers"]
    n_returned_ans = len(returned_ans)
    n_contexts_w_answers = len(rs["correct_contexts"])
    print("Returned Answers:", returned_ans)
    print(
        f"[Recall: {rs['recall'] * 100.0 :0.2f}%] {n_returned_ans} out of {rs['n_gt_ans']} in context list"
    )
    print(
        f"[Precision: {rs['precision'] * 100.0 :0.2f}%] {n_contexts_w_answers} out of {rs['n_retrieved_ctx']} contexts contained an answer"
    )


# Calculate aggregate metrics on a dataset (currently can't limit the @k to anything)
def evaluate_dataset(dataset, k=None):
    recalls = []
    precisions = []
    f1s = []
    for data_point in dataset:
        ctxs = data_point["ctxs"][:k] if k is not None else data_point["ctxs"]
        rs = correct_answers_context_list(
            context_list=ctxs,
            answer_llist=list(data_point["ans_mappings"].values()),
        )
        recalls.append(rs["recall"])
        precisions.append(rs["precision"])
        f1 = 0
        f1_denom = rs["recall"] + rs["precision"]
        if f1_denom > 0:
            f1 = 2 * rs["recall"] * rs["precision"] / f1_denom
        f1s.append(f1)
    return {
        "avg_recall": sum(recalls) / len(recalls),
        "avg_precision": sum(precisions) / len(precisions),
        "avg_f1": sum(f1s) / len(f1s),
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
    }


# Get relevant metrics for a single question/answer set + retrieved results
# Expects a context to have a 'text' key and the answer_llist to be a list of
#   lists of answers.
def correct_answers_context_list(context_list, answer_llist):
    correct_answers = set()
    correct_contexts = set()
    for ci, c in enumerate(context_list):
        ctext = c["text"]
        cbools = answers_in_context(context=ctext, answer_llist=answer_llist)
        if any(cbools):
            correct_contexts.add(ci)
        corr_ansids = [ai for ai, ca in enumerate(cbools) if ca]
        correct_answers.update(corr_ansids)
    n_ans = len(answer_llist)
    n_ctx = len(context_list)
    n_corr_ans = len(correct_answers)
    n_corr_ctx = len(correct_contexts)
    return {
        "correct_answers": [answer_llist[ai][0] for ai in correct_answers],
        "correct_contexts": list(correct_contexts),
        "recall": n_corr_ans / n_ans,
        "precision_unique": n_ans / n_ctx,
        "precision": n_corr_ctx / n_ctx,
        "n_gt_ans": n_ans,
        "n_retrieved_ctx": n_ctx,
    }


# Check whether any alias of the answer is in the context
# - takes the context and a list of alias lists (one list per answer)
# - returns a True/False presence list for (any alias of) all answers
def answers_in_context(context, answer_llist):
    ans_in_context = []
    for alist in answer_llist:
        a_in_context = False
        for a in alist:
            if a in context:
                a_in_context = True
                break
        ans_in_context.append(a_in_context)
    return ans_in_context


# Returns list of the index of the context that contains each answer (or None)
# Expects retrieval list of the form:
#    [{"question": ...,
#      "contexts": { # (or "ctxs")
#         ctx_idx: {"text": "..."},
#      ...},
#     },
#    ...]
# Expects ground truth: [(question, [answer_aliases, ...]), ...]
#
# Returns list of (question, [answer_inds])
def get_index_of_context_with_answers(retrieval_list, ground_truth):
    q_not_in_gt = []
    indices_of_answers = []
    for dev_idx in range(len(retrieval_list)):
        retrieved = retrieval_list[dev_idx]
        if retrieved["question"] not in ground_truth:
            q_not_in_gt.append(retrieved["question"])
            continue
        gt_ans = ground_truth[retrieved["question"]]
        index_true = [None for _ in range(len(gt_ans))]
        ctx_key = "contexts" if "contexts" in retrieved else "ctxs"
        for ctx_idx in range(len(retrieved[ctx_key])):
            a_in_c = answers_in_context(retrieved[ctx_key][ctx_idx]["text"], gt_ans)
            for a_idx, a in enumerate(a_in_c):
                if a and index_true[a_idx] is None:
                    index_true[a_idx] = ctx_idx
            if all([it is not None for it in index_true]):
                break
        indices_of_answers.append((retrieved["question"], index_true))
    return indices_of_answers, q_not_in_gt


# Indices of Answers list to Rank at k
# Returns:
# - the average rank at k
# - each questions rank p at k
# - each questions recall at k
def ioa_to_rankPatK(indices_of_answers, k):
    num_at_k = []
    rankP_at_k = []
    for dev_idx in range(len(indices_of_answers)):
        num_ans = len(indices_of_answers[dev_idx][1])
        num_below_k = 0
        for ioa in indices_of_answers[dev_idx][1]:
            if ioa is not None and ioa < k:
                num_below_k += 1
        p_below_k = num_below_k / num_ans
        num_at_k.append(num_below_k)
        rankP_at_k.append(p_below_k)
    return sum(rankP_at_k) / len(rankP_at_k), rankP_at_k, num_at_k
