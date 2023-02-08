#!/bin/bash
#SBATCH --job-name=test_job_arrays
#SBATCH --array=0-20%5
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/sbatch_test/%A_j%a.out
#SBATCH --error=/scratch/ddr8143/sbatch_test/%A_j%a.err
#SBATCH --export=ALL
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:rtx8000:0
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate multiqa

python /scratch/ddr8143/repos/multiqa_utils/sbatch_test.py
"
