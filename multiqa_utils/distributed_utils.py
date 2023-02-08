import json
import os
from tqdm import tqdm
import math
import shutil
import glob

import multiqa_utils.general_utils as gu
import multiqa_utils.wikipedia_utils as wu

def split_list_to_jobs(
    job_id,
    total_num_jobs,
    full_list,
):
    total_elem = len(full_list)
    num_strs_each = math.ceil(total_elem / total_num_jobs)
    start_id = num_strs_each * job_id
    end_id = num_strs_each * (job_id+1)
    print(f">> This job gets from {start_id} to {end_id} of full list length {total_elem} bc its job {job_id} of {total_num_jobs}")
    return full_list[start_id:end_id]


def distributed_build_str2wikipage_cache(
    path_args,
    job_id,
    total_num_jobs,
    all_strs_to_add,
    force=False,
    use_tqdm=False,
    write_every=None,
):
    # First setup this job specifically
    strs_to_add = split_list_to_jobs(job_id, total_num_jobs, all_strs_to_add)    
    wu.build_str2wikipage_cache(
        path_args,
        strs_to_add=strs_to_add,
        force=force,
        use_tqdm=use_tqdm,
        write_every=write_every,
        suffix=f"__job{job_id}"
    )


def aggregate_checkpoint_dicts(
    base_path,
    remove_processed=False,
):
    all_files = [f for f in glob.glob(f"{base_path}*") if "__old" not in f]
    if len(all_files) < 2:
        print(f">> Only one file, skipping aggregation")
        return
    
    print(f">> Aggregating all {len(all_files)} versions of:", base_path)
    full_dict = {}
    for f in all_files:
        file_dict = json.load(open(f))
        full_dict.update(file_dict)
    print(">> Length of final dict:", len(full_dict))
    gu.checkpoint_json(data=full_dict, path=base_path)
    if remove_processed:
        to_remove = [f for f in all_files if f != base_path]
        backups = [gu.get_backup_path(f) for f in to_remove]
        to_remove = to_remove + [bf for bf in backups if os.path.exists(bf)]
        for f in to_remove:
            os.remove(f)
        print(">> Intermediate files have been removed")
    