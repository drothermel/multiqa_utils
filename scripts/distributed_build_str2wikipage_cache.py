#!/ext3/miniconda3/envs/multiqa/bin/python -u

import sys
import os
import json

import multiqa_utils.general_utils as gu
import multiqa_utils.wikipedia_utils as wu
import multiqa_utils.distributed_utils as du

def run_build(job_id, total_num_jobs):
    path_args = gu.current_default_path_args()
    print(">> Path args:", flush=True)
    for k, v in vars(path_args).items():
        print(k, v, flush=True)
    
    all_titles = wu.build_gt_wikititle_set(path_args, force=False)
    curr_cache = wu.get_initial_str2wikipage_cache(all_titles, path_args, force=False)
    strs_to_add = json.load(open(path_args.strs_for_cache_path))
    du.distributed_build_str2wikipage_cache(
        path_args,
        job_id=job_id,
        total_num_jobs=total_num_jobs,
        all_strs_to_add=strs_to_add,
        verbose=True,
        curr_cache=curr_cache,
        write_every=5000,
        use_tqdm=False,    
    )
    print(f">> Finished running job: {job_id}/{total_num_jobs}")
    
    
if __name__ == "__main__":
    # Read in our env variables
    
    if "SLURM_ARRAY_TASK_COUNT" not in os.environ or "SLURM_ARRAY_TASK_ID" not in os.environ:
        print("Something is missing!")
        print("SLURM_ARRAY_TASK_COUNT", "SLURM_ARRAY_TASK_COUNT" not in os.environ)
        print("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_ID" not in os.environ)
        sys.exit()
        
    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    #total_num_jobs = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    total_num_jobs = 10
    print(f">> Running: {job_id} / {total_num_jobs}", flush=True)
    run_build(job_id, total_num_jobs)
    sys.exit()