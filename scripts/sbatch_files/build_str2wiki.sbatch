#!/bin/bash
#SBATCH --job-name=dist_build_str2wikipage_400_v0
#SBATCH --array=0-9
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/multiqa/slurm_logs/%x_%A_j%a.out
#SBATCH --error=/scratch/ddr8143//multiqa/slurm_logs/%x_%A_j%a.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:0
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate multiqa

python /scratch/ddr8143/repos/multiqa_utils/scripts/distributed_build_str2wikipage_cache.py
"
