


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