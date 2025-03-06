#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -o qlogs/
#$ -e qlogs/

#$ -l rocky

#$ -l cluster=andrena
#$ -l h_vmem=7.5G
#$ -pe smp 12

# -l node_type=rdg
# -l gpuhighmem
# -l h_vmem=20G
# -pe smp 12

#$ -t 7-8

EXPERIMENT=$(sed -n "${SGE_TASK_ID}p" jobs/train/surge/experiments.txt)

rm -rf ~/.triton/cache
mamba activate perm
module load gcc
python src/train.py experiment=$EXPERIMENT \
  seed=999
