import numpy as np
import matplotlib.pyplot as plt

from utils.util_classes import Metrics
import multiqa_utils.string_utils as su


# --------- Building Blocks ----------- #


def plot_pr_data(p, r, t, nc, box=True, psize=5):
    if box:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(2 * psize, 2 * psize)
        )
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(4 * psize, 1 * psize))
    fig.suptitle("Horizontally stacked subplots")
    ax1.plot(r, p, "-o")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax2.plot(t, r, "-o")
    ax2.set_xlabel("threshold")
    ax2.set_ylabel("recall")
    ax3.plot(t, p, "-o")
    ax3.set_xlabel("threshold")
    ax3.set_ylabel("precision")
    ax4.plot(t, nc, "-o")
    ax4.set_xlabel("threshold")
    ax4.set_ylabel("num_cands")


def add_accuracy_metrics(
    recalls,
    precisions,
    f1s,
    metrics=None,
    prefix='',
):
    if prefix != '':
        prefix += '_'
    if metrics is None:
        metrics = Metrics()

    perf_precision = [float(v == 1.0) for v in precisions]
    perf_recall = [float(v == 1.0) for v in recalls]
    perf_f1 = [float(v == 1.0) for v in f1s]
    metrics.increment_val(
        f'{prefix}avg_perfect_recall', sum(perf_precision) / len(perf_precision)
    )
    metrics.increment_val(
        f'{prefix}avg_perfect_precision', sum(perf_recall) / len(perf_recall)
    )
    metrics.increment_val(f'{prefix}avg_exact_correct', sum(perf_f1) / len(perf_f1))
    return metrics


def calc_precision(num_preds, num_preds_in_gt):
    if num_preds == 0:
        return 0.0
    return num_preds_in_gt * 1.0 / num_preds


# This shouldn't really happen, but will in Romqa depending on how you
# load the data.
def calc_recall(num_gt_ans, gt_inds_found):
    if num_gt_ans == 0:
        return 1.0
    return len(gt_inds_found) * 1.0 / num_gt_ans


def calc_f1(precision, recall):
    if (precision == 0.0 and recall == 0.0) or precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calc_precision_recall_f1_nc_simple(
    num_preds,
    num_gt_ans,
    num_preds_in_gt,
    gt_inds_found,
    metrics=None,
    prefix=None,
    per=None,
):
    results = {}
    results['precision'] = calc_precision(num_preds, num_preds_in_gt)
    results['recall'] = calc_recall(num_gt_ans, gt_inds_found)
    results['f1'] = calc_f1(results['precision'], results['recall'])
    results['num_cands'] = num_preds

    # Return results for a single element
    if metrics is None:
        return results

    # Add results to lists when calculating across a dataset
    pref = '' if prefix is None else f'{prefix}_'
    for k, v in results.items():
        metrics.add_to_lists_per(f'{pref}{k}', per, v)
    return metrics


# Per-question function
# Inputs:
#   gt_sets_list - [{ans_1_alias_1, ans_1_alias_2}, {ans_2_alias_1}, ...]
#   preds_list - either [pred, ...] or [(pred, ..), ..]
#                returns the normed_preds only in the same order
#
# Return normed_gt_sets_list, normed_preds_list, norm_cache
#   with norms applied or not according to flags.
def elem_apply_norms_to_preds_gt(
    gt_sets_list,
    preds_list,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
    norm_cache={},
):
    normed_gt_sets_list = gt_sets_list
    if len(norm_fxns) > 0 and apply_to_gt:
        normed_gt_sets_list = []
        for gt_set in gt_sets_list:
            normed_gt_set = set()
            for gt_ans in gt_set:
                if gt_ans in norm_cache:
                    norm_gt_ans = norm_cache[gt_ans]
                else:
                    norm_gt_ans = su.apply_norms(gt_ans, norm_fxns)
                    norm_cache[gt_ans] = norm_gt_ans
                normed_gt_set.add(norm_gt_ans)
            normed_gt_sets_list.append(normed_gt_set)

    normed_preds_list = preds_list
    if len(norm_fxns) > 0 and apply_to_pred:
        normed_preds_list = []
        for pred_or_pred_tuple in preds_list:
            # Note pred might be "pred" or ("pred", "score", ...)
            pred = pred_or_pred_tuple
            if isinstance(pred_or_pred_tuple, tuple):
                pred = pred_or_pred_tuple[0]

            if pred in norm_cache:
                norm_pred = norm_cache[pred]
            else:
                norm_pred = su.apply_norms(pred, norm_fxns)
                norm_cache[pred] = norm_pred
            normed_preds_list.append(norm_pred)

    return normed_gt_sets_list, normed_preds_list, norm_cache


# --------- PR Calcs Without Scores ----------- #

# Per-question function
# Inputs:
#   gt_ans_sets_list - [{ans_1_alias_1, ans_1_alias_2}, {ans_2_alias_1}, ...]
#   preds_list - either [pred, ...]
#
# Return metrics_dict (either dict or Metrics) and norm_cache
def elem_calc_precision_recall_f1_no_scores(
    gt_sets_list,
    preds_list,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
    norm_cache=dict(),
    metrics=None,
    prefix=None,
    per=None,
):
    # First get the normed verisons
    (normed_gt_sets_list, normed_preds_list, norm_cache) = elem_apply_norms_to_preds_gt(
        gt_sets_list,
        preds_list,
        norm_fxns=norm_fxns,
        apply_to_gt=apply_to_gt,
        apply_to_pred=apply_to_pred,
        norm_cache=norm_cache,
    )

    # Then calculate the stats
    num_preds = len(preds_list)
    num_gt_ans = len(gt_sets_list)
    num_preds_in_gt = 0
    gt_inds_found = set()
    for pred_ind, pred in enumerate(preds_list):
        normed_pred = normed_preds_list[pred_ind]
        pred_in_gt = False
        for gt_ind, gt_ans_set in enumerate(normed_gt_sets_list):
            if normed_pred in gt_ans_set:
                gt_inds_found.add(gt_ind)
                pred_in_gt = True

        if pred_in_gt:
            num_preds_in_gt += 1

    results = calc_precision_recall_f1_nc_simple(
        num_preds,
        num_gt_ans,
        num_preds_in_gt,
        gt_inds_found,
        metrics=metrics,
        prefix=prefix,
        per=per,
    )
    return results, norm_cache


def dataset_calc_precision_recall_f1_no_scores(
    gt_sets_list_of_lists,
    preds_list_of_lists,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
    norm_cache={},
    metrics=None,
    prefix=None,
    per=None,
):
    if metrics is None:
        metrics = Metrics()

    for q_ind, gt_sets_list in enumerate(gt_sets_list_of_lists):
        metrics, norm_cache = elem_calc_precision_recall_f1_no_scores(
            gt_sets_list,
            preds_list_of_lists[q_ind],
            norm_fxns=norm_fxns,
            apply_to_gt=apply_to_gt,
            apply_to_pred=apply_to_pred,
            norm_cache=norm_cache,
            metrics=metrics,
            prefix=prefix,
            per=per,
        )
    return metrics, norm_cache


# --------- PR Calcs With Scores ----------- #

# Per-question function
# Inputs:
#   gt_sets_list - [{ans_1_alias_1, ans_1_alias_2}, {ans_2_alias_1}, ...]
#   preds_w_scores_list - either [(pred, score), ..]
#
# Return: [(pred_str, score, set(matching_ans_inds)), ...]
def elem_gtset_predscores_add_matchans(
    gt_sets_list,
    preds_w_scores_list,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
    norm_cache={},
):
    pred_score_matchans = []

    # First get the appropriately normed lists
    (normed_gt_sets_list, normed_preds_list, norm_cache) = elem_apply_norms_to_preds_gt(
        gt_sets_list,
        preds_w_scores_list,
        norm_fxns,
        apply_to_gt,
        apply_to_pred,
        norm_cache,
    )

    # Then get all matches between preds and gt
    for pred_i, (pred, score) in enumerate(preds_w_scores_list):
        normed_pred = normed_preds_list[pred_i]  # will obey apply_to_pred
        matching_gt_inds = set()
        for gt_ind, gt_alias_set in enumerate(normed_gt_sets_list):
            if normed_pred in gt_alias_set:
                matching_gt_inds.add(gt_ind)
        pred_score_matchans.append((pred, score, matching_gt_inds))
    return pred_score_matchans, norm_cache


# For a full datasets
# Inputs:
# - gt_sets_list_of_lists = [
#       [{ans_1_alias_1, ans_1_alias_2}, {ans_2_alias_1}, ...],
#       ...,
# ]
# - preds_w_scores_list_of_lists = [
#       [(pred, score), (pred, score), ...],
#       ...,
# ]
#
# Return: [
#     [(pred_str, score, set(matching_ans_inds)), ...],
#     ...,
# ]
def dataset_gtset_predscores_add_matchans(
    gt_sets_list_of_lists,
    preds_w_scores_list_of_lists,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
):
    norm_cache = {}
    pred_score_matchans_list_of_lists = []
    for i in range(len(gt_sets_list_of_lists)):
        pred_score_matchans_list, norm_cache = elem_gtset_predscores_add_matchans(
            gt_sets_list_of_lists[i],
            preds_w_scores_list_of_lists[i],
            norm_fxns,
            apply_to_gt,
            apply_to_pred,
            norm_cache,
        )
        pred_score_matchans_list_of_lists.append(pred_score_matchans_list)
    return pred_score_matchans_list_of_lists


# pred_score_matchans:
#     [(pred, score, set(gt_ind_match1, gt_ind_match2)), ...]
def elem_pr_curve_linspace(
    pred_score_matchans,
    num_gt_ans,
    num_samples=100,
    thr_range=None,
    metrics=None,
    prefix=None,
    per=None,
):
    # Sort in preparation for PR curve calc and get thr_range
    rev_sort_psm = sorted(pred_score_matchans, reverse=True, key=lambda pr, sc, ma: sc)
    eps = 0.00001
    max_thr = rev_sort_psm[0][1] + eps if thr_range is None else thr_range[1]
    min_thr = rev_sort_psm[-1][1] - eps if thr_range is None else thr_range[0]

    thrs = np.linspace(max_thr, min_thr, num_samples)
    thr_ind = 0

    # Record the stats in a Metrics
    if metrics is None:
        metrics = Metrics()
    stats_to_calc = ['precision', 'recall', 'f1', 'num_cands']
    key_names = [metrics.get_val_name(k, per=per, prefix=prefix) for k in stats_to_calc]

    # If the max_thr > max_score:
    #     -> nothing is predicted for first thr
    #     -> all stats are 0.0
    if max_thr > rev_sort_psm[0][1]:
        for key in key_names:
            metrics.add_to_lists_per(key, per, 0.0)
        thr_ind = 1

    num_preds_in_gt = 0
    gt_inds_found = set()
    num_scores_above_thr = 0
    for pr, sc, ma in rev_sort_psm:
        # If the score is less than the threshold, record precision
        # recall and num_cands for this threshold and update threshold
        if sc < thrs[thr_ind]:
            metrics = calc_precision_recall_f1_nc_simple(
                num_scores_above_thr,
                num_gt_ans,
                num_preds_in_gt,
                gt_inds_found,
                metrics=metrics,
                prefix=prefix,
                per=per,
            )
            while sc < thrs[thr_ind]:
                for key in key_names:
                    metrics.repeat_prev_lists_per(key, per)
                thr_ind += 1
                if thr_ind >= len(thrs):
                    break

        # Now sc >= thrs[thr_ind] so we this (pr, sc, ma) is newly predicted
        num_scores_above_thr += 1

        # This prediction has at least one matching answer
        if len(ma) > 0:
            num_preds_in_gt += 1

            # Add the matching answers to the found gt ans inds
            gt_inds_found.update(ma)

        # Don't add to the metrics because we only record metrics at fixed
        # values of threshold and there might be more preds above this thr

    # Now that all the predictions have been predicted at the current thr_ind
    # we need to keep the same prfn for the rest of the thresholds
    while thr_ind < len(thrs):
        for key in key_names:
            metrics.repeat_prev_lists_per(key, per)
        thr_ind += 1

    # Add thresholds to the metrics
    thr_name = metrics.get_val_name('thresholds', prefix=prefix)
    metrics.lists_per[(thr_name, per)] = thrs

    return metrics


# Inputs:
# - preds_w_scores_list_of_lists = [
#       [(pred, score), (pred, score), ...],
#       ...,
# ]
# - gt_ans_sets_list_of_lists = [
#       [{ans_1_alias_1, ans_1_alias_2}, {ans_2_alias_1}, ...],
#       ...,
# ]
def dataset_pr_curve_linspace(
    preds_w_scores_list_of_lists,
    gt_ans_sets_list_of_lists,
    num_samples=100,
    norm_fxns=[],
    apply_to_gt=True,
    apply_to_pred=True,
    thr_range=None,
    metrics=None,
    prefix=None,
    per=None,
):
    # First conver the scores and gt_ans into list of
    # pred_score_matchans and num_gt_ans
    num_gt_ans = [len(gtas_list) for gtas_list in gt_ans_sets_list_of_lists]
    pred_score_matchans_list_of_lists = dataset_gtset_predscores_add_matchans(
        gt_ans_sets_list_of_lists,
        preds_w_scores_list_of_lists,
        norm_fxns,
        apply_to_gt,
        apply_to_pred,
    )

    # Then choose the threshold ranges
    eps = 0.00001
    if thr_range is not None:
        min_thr = thr_range[0]
        max_thr = thr_range[1]
    else:
        min_thr = None
        max_trh = None
        for pws_list in preds_w_scores_list_of_lists:
            mins = min([ps[1] for ps in pws_list]) - eps
            maxs = max([ps[1] for ps in pws_list]) + eps
            if min_thr is None or min_thr > mins:
                min_thr = mins
            if max_trh is None or max_thr < maxs:
                max_thr = maxs

    # Setup the metrics to record the values and count
    if metrics is None:
        metrics = Metrics()
    stats_to_calc = ['precision', 'recall', 'f1', 'num_cands']
    key_names = [metrics.get_val_name(k, per=per, prefix=prefix) for k in stats_to_calc]
    count_key = metrics.get_val_name('count', per, prefix=prefix)
    metrics.increment_val(count_key)

    # Then iterate through getting the linspaced values
    for i in range(len(num_gt_ans)):
        metrics = elem_pr_curve_linspace(
            pred_score_matchans_list_of_lists[i],
            num_gt_ans[i],
            num_samples=num_samples,
            thr_range=[min_thr, max_thr],
            metrics=metrics,
            prefix=prefix,
            per=per,
        )
        metrics.increment_val(count_key)

        # Either initialize the relevant array or sum into it
        for key in key_names:
            metrics.convert_list_to_array(key, per)

    # Normalize the sum arrays by the count
    for key in key_names:
        metrics.norm_arrays_per(key, per, metrics.vals[count_key])
    return metrics


"""
# Everything below here is old

# More recent PR Curve creation
def make_accurate_pr_curve(preds_list, gt_missing):
    total_missing_gt = sum(gt_missing)  # each elem is num_missing_in_question

    # Get total number unique gt and the remaining gt correctly predicted info
    num_unique_found_gt = len(
        [p["gt_id"] for p in preds_list if p["gt_id"] is not None]
    )
    total_num_gt = total_missing_gt + num_unique_found_gt
    logging.info(f">> Calculating PR Curve: {name}")
    logging.info(f">> - total_missing_gt: {total_missing_gt:,}")
    logging.info(f">> - unique_found_gt: {num_unique_found_gt:,}")
    logging.info(f">> - total_num_gt: {total_num_gt:,}")

    # gt_rem: { gt_id: set(pred_id, ...), }
    gt_remaining = defaultdict(set)
    for p in preds_list:
        if p["gt_id"] is not None:
            gt_remaining[p["gt_id"]].add(p["pred_id"])

    sorted_pr_data = sorted(
        preds_list,
        key=lambda d: d["score"],
    )
    ep = 0.00001
    min_score = sorted_pr_data[0]["score"]
    num_remaining_corr_pred = sum([d["label"] for d in sorted_pr_data])
    num_remaining = len(sorted_pr_data)
    num_remaining_gt = len(gt_remaining)
    ps = [num_remaining_corr_pred / num_remaining]
    rs = [num_remaining_gt / total_num_gt]
    ths = [min_score - ep]
    logging.info(f">> - min_score: {min_score:0.3f}")
    logging.info(f">> - max_num_corr_pred: {int(num_remaining_corr_pred):,}")
    logging.info(f">> - num_total_preds: {int(num_remaining):,}")
    logging.info(f">> - num_total_gt_remaining: {int(num_remaining_gt):,}")
    logging.info(f">> - min_thr: {ths[0]:0.3f}")
    logging.info(f">> - min_thr_precision: {ps[0]:0.4f}")
    logging.info(f">> - min_thr_recall: {rs[0]:0.3f}")
    for i in range(len(sorted_pr_data) - 1):
        new_pt = sorted_pr_data[i + 1]
        new_thr = new_pt["score"] - ep
        num_remaining -= 1
        lost_label = sorted_pr_data[i]["label"]
        if lost_label == 1:
            num_remaining_corr_pred -= 1
            lost_gt_id = sorted_pr_data[i]["gt_id"]
            lost_pred_id = sorted_pr_data[i]["pred_id"]
            gt_remaining[lost_gt_id].remove(lost_pred_id)
            if len(gt_remaining[lost_gt_id]) == 0:
                del gt_remaining[lost_gt_id]
        if num_remaining == 0:
            break
        ps.append(num_remaining_corr_pred / num_remaining)
        rs.append(len(gt_remaining) / total_num_gt)
        ths.append(new_thr)
    return ps, rs, ths


# And some more!


def pr_curve_qdata(ents, scores, qdata):
    dt = su.get_detokenizer()
    ans = set([wu.qmp_norm(dt, qa["answer_text"]) for qa in qdata["answer_list"]])

    tot_ans_found = len(ans - set(ents))
    tot_ans = len(ans)

    epsilon = 0.000001
    precision = []
    recall = []
    num_cands = []
    threshold = []

    curr_tp = 0
    curr_p = 0
    for s, e in sorted(list(zip(scores, ents)), reverse=True):
        curr_p += 1
        if e in ans:
            curr_tp += 1

        p_over_thr = curr_tp
        precision.append(curr_tp * 1.0 / curr_p)
        recall.append(curr_tp * 1.0 / tot_ans)
        threshold.append(s - epsilon)
        num_cands.append(curr_p)
    return precision, recall, threshold, num_cands


# def pr_curve_qdata_linspace(ents, scores, qdata, num_samples=100):
#    ans = set([wu.qmp_norm(qa["answer_text"]) for qa in qdata["answer_list"]])
#    tot_ans_found = len(ans - set(ents))
#    tot_ans = len(ans)
# Assumes uniqueness of ents
def pr_curve_qdata_linspace(ents, scores, labels, tot_ans, num_samples=100):
    thr_samples = np.linspace(1.0, 0.0, num_samples)
    curr_tp = 0
    curr_p = 0
    precision = [0.0]  # values for thr = 1.0
    recall = [0.0]
    threshold = [1.0]
    num_cands = [0.0]

    thr_ind = 1
    thr = thr_samples[thr_ind]
    for s, e, l in sorted(list(zip(scores, ents, labels)), reverse=True):
        # print(s, e, l)
        while s < thr:
            precision.append(curr_tp * 1.0 / curr_p if curr_p != 0.0 else 0.0)
            recall.append(curr_tp * 1.0 / tot_ans)
            threshold.append(thr)
            num_cands.append(curr_p)
            thr_ind += 1
            if thr_ind >= len(thr_samples):
                break
            thr = thr_samples[thr_ind]
        if thr_ind >= len(thr_samples):
            break
        curr_p += 1
        if l == 1:
            curr_tp += 1

    while thr_ind < len(thr_samples):
        precision.append(precision[-1])
        recall.append(recall[-1])
        threshold.append(thr_samples[thr_ind])
        num_cands.append(num_cands[-1])
        thr_ind += 1
    return precision, recall, threshold, num_cands


# Expects 'candidates', 'filtering_scores'
# def avg_qdata_pr_curves(scoring_data, qdata, num_samples, scores, fscore=0):
def avg_qdata_pr_curves(scoring_data, num_samples):
    allp = None
    allr = None
    allt = None
    allnc = None
    count = 0
    for qid, qd in scoring_data.items():
        # print(qid, count)
        ets = qd["candidates"]
        scs = qd["filtering_scores"]
        ls = qd["labels"]
        nca = qd["num_correct_answers"]
        p, r, t, nc = pr_curve_qdata_linspace(ets, scs, ls, nca, num_samples)
        count += 1
        if allp is None:
            allp = np.array(p)
            allr = np.array(r)
            allt = np.array(t)
            allnc = np.array(nc)
        else:
            allp = allp + np.array(p)
            allr = allr + np.array(r)
            allt = allt + np.array(t)
            allnc = allnc + np.array(nc)
    max_recall = allr[-1] *100.0 / count
    ncands = int(allnc[-1] / count)
    print(
        f">> Max recall: {max_recall:0.4f}% with {ncands:,} cands"
    )
    return allp / count, allr / count, allt / count, allnc / count


def plot_pr_data_together(
    plist,
    rlist,
    tlist,
    nclist,
    labels,
    bavgr,
    bavgp,
    bavgnc,
    gptavgr,
    gptavgp,
    gptavgnc,
):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    # fig.suptitle(
    #   "Comparison of Precision/Recall Values for Different Scoring Methods"
    # )
    for i, p in enumerate(plist):
        ax1.plot(rlist[i], p, "-", label=labels[i])
        ax2.plot(tlist[i], rlist[i], "-", label=labels[i])
        ax3.plot(tlist[i], p, "-", label=labels[i])
        ax4.plot(tlist[i], nclist[i], "-", label=labels[i])

    ax1.plot([bavgr], [bavgp], "o", markersize=15, label="Baseline")
    ax1.plot([gptavgr], [gptavgp], "o", markersize=15, label="GPT3")
    ax1.set_title("Precision vs. Recall", fontsize=14)
    ax2.set_title("Recall vs. Threshold", fontsize=14)
    ax3.set_title("Precision vs. Threshold", fontsize=14)
    ax4.set_title("Num Candidates vs. Threshold", fontsize=14)

    thr_samples = np.linspace(0.0, 1.0, 100)
    ax2.plot(thr_samples, [bavgr for _ in range(len(thr_samples))])
    ax2.plot(thr_samples, [gptavgr for _ in range(len(thr_samples))])
    ax3.plot(thr_samples, [bavgp for _ in range(len(thr_samples))])
    ax3.plot(thr_samples, [gptavgp for _ in range(len(thr_samples))])
    ax4.plot(thr_samples, [bavgnc for _ in range(len(thr_samples))])
    ax4.plot(thr_samples, [gptavgnc for _ in range(len(thr_samples))])

    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.legend(fontsize=14)
    ax2.set_xlabel("threshold")
    ax2.set_ylabel("recall")
    ax3.set_xlabel("threshold")
    ax3.set_ylabel("precision")
    ax4.set_xlabel("threshold")
    ax4.set_ylabel("num candidates")
    plt.tight_layout()
"""
