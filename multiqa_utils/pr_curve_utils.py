## Note that thises all might be wrong, or correct, def don't just assume they work!
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt

import multiqa_utils.string_utils as tu

def plot_pr_data(p, r, t, nc, box=True, psize=5):
    if box:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(2*psize, 2*psize)
        )
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            1, 4, figsize=(4*psize, 1 * psize)
        )
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

def calc_precision_recall_f1_nc_simple(
    num_preds,
    num_gt_ans,
    num_preds_in_gt,
    gt_inds_found,
    num_scores_above_thr,
):
    if len(preds_w_scores) == 0:
        precision = 0.0
    else:
        precision = num_preds_in_gt * 1.0 / num_preds

    if len(gt_ans_sets_list) == 0:
        recall = 1.0
    else:
        recall = len(gt_inds_found) * 1.0 / num_gt_ans

    f1 = 2 * precision * recall / (precision + recall)
    num_cands = num_scores_above_thr
    return precision, recall, f1, num_cands
    


# For a single question:
# - list of gt answers, alias set for each
# - list or predictions [(pred_str, score), ...]
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

    # Normalize gt if required
    normed_gt_sets = gt_sets_list
    if len(norm_fxns) > 0 and apply_to_gt:
        normed_gt_sets = []
        for gt_set in gt_sets_list:
            normed_gt_sets.append(set())
            for gt_ans in gt_set:
                if gt_ans in norm_cache:
                    normed_ans = norm_cache[gt_ans]
                else:
                    normed_ans = su.apply_norms(gt_ans, norm_fxns)
                    norm_cache[gt_ans] = normed_ans
                normed_gt_sets[-1].add(normed_ans)

    # Add normalize preds to norm_cache if required
    if len(norm_fxns) > 0 and apply_to_pred:
        for pred, score in preds_w_scores_list:
            if pred in not norm_cache:
                normed_pred = su.apply_norms(pred, norm_fxns)
                norm_cache[pred] = normed_pred

    # Then get all matches between preds and gt
    for pred, score in preds_w_scores_list:
        normed_pred = norm_cache[pred]
        matching_gt_inds = set()
        for gt_ind, gt_alias_set in enumerate(normed_gt_sets):
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
    for i in range(len(gt_sets_list)):
        pred_score_matchans_list, norm_cache = elem_gtset_predscores_add_matchans(
            gt_sets_list_of_lists[i],
            preds_w_scores_list_of_lists[i],
            norm_fxns,
            apply_to_gt,
            apply_to_pred,
            norm_cache,
        )
        pred_score_matchans_list_of_lists.append(
            pred_score_matchans_list
        )
    return pred_score_matchans_list_of_lists


# pred_score_matchans:
#     [(pred, score, set(gt_ind_match1, gt_ind_match2)), ...]
def elem_pr_curve_linspace(
    pred_score_matchans,
    num_gt_ans,
    num_samples=100,
    thr_range=None,
):
    # Sort in preparation for PR curve calc and get thr_range
    num_preds = len(pred_score_matchans)
    rev_sort_psm = sorted(
        pred_score_matchans, reverse=True, key=lambda pr, sc, ma: sc
    )
    eps = 0.00001
    max_thr = rev_sort_psm[0][1] + eps if thr_range is None else thr_range[1]
    min_thr = rev_sort_psm[-1][1] - eps if thr_range is None else thr_range[0]

    thrs = np.linspace(max_thr, min_thr, num_samples)
    thr_ind = 0
    precisions = []
    recalls = []
    f1s = []
    num_cands = []

    # If the max_thr > max_score:
    #     -> nothing is predicted for first thr
    #     recall = 0, nothing predicted
    #     precision = 0, nothing predicted
    #     nc = 0, nothing predicted
    # However, if max_thr <= max_score, this isn't true
    if max_thr > rev_sort_psm[0][1]:
        precisions.append(0.0)
        recalls.append(0.0)
        f1s.append(0.0)
        num_cands.append(0.0)
        thr_ind = 1

    num_preds_in_gt = 0
    gt_inds_found = set()
    num_scores_above_thr = 0
    for pr, sc, ma in rev_sort_psm:
        # If the score is less than the threshold, record precision
        # recall and num_cands for this threshold and update threshold
        if sc < thrs[thr_ind]:
            pr, rc, f1, nc = calc_precision_recall_f1_nc_simple(
                num_preds,
                num_gt_ans,
                num_preds_in_gt,
                gt_inds_found,
                num_scores_above_thr,
            )
            while sc < thrs[thr_ind]:
                precisions.append(pr)
                recalls.append(rc)
                f1s.append(f1)
                num_cands.append(nc)
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
        precisions.append(precisions[-1])
        recalls.append(recalls[-1])
        f1s.append(f1s[-1])
        num_cands.append(num_cands[-1])
        thr_ind += 1

    return {
        'thresholds': thrs,
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'num_cands': num_cands,
    }


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
        for pws_list in preds_w_scores_list_of_list:
            mins = min([ps[1] for ps in pws_list]) - eps
            maxs = max([ps[1] for ps in pws_list]) + eps
            if min_thr is None or min_thr > mins:
                min_thr = mins
            if max_trh is None or max_thr < maxs:
                max_thr = maxs

    # Then iterate through getting the linspaced values
    avg_values = {}
    count = 0
    for i in range(len(num_gt_ans)):
        res = elem_pr_curve_linspace(
            pred_score_matchans_list_of_lists[i],
            num_gt_ans[i],
            num_samples=num_samples,
            thr_range=[min_thr, max_thr],
        )
        for k, v in res.items():
            varr = np.array(v)
            count += 1
            if k not in avg_values:
                avg_values[k] = varr
            else:
                avg_values[k] += varr
    return {k: varr / count for k, varr in avg_values.items()}



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
    print(
        f">> Max recall: {allr[-1] *100.0 / count:0.4f}% with {int(allnc[-1] / count):,} cands"
    )
    return allp / count, allr / count, allt / count, allnc / count


def plot_pr_data(p, r, t, nc):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
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
    # fig.suptitle("Comparison of Precision/Recall Values for Different Scoring Methods")
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
