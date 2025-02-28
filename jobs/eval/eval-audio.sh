#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=4G
#$ -pe smp 16
#$ -l centos
#$ -l node_type=ddy
#$ -cwd
#$ -j y
#$ -o dlogs/
#$ -e dlogs/
#$ -t 1-8

JOB_DIR=$(sed -n "${SGE_TASK_ID}p" jobs/eval/eval-jobs.txt)

AUDIO_DIR=${JOB_DIR}/audio
METRIC_DIR=${JOB_DIR}/metrics

mkdir -p $METRIC_DIR

echo Running predict job on $JOB_DIR.
echo Audio folder is $AUDIO_DIR
echo Metric folder is $METRIC_DIR

module load gcc
mamba activate perm
python scripts/compute_audio_metrics.py -w 16 -- $AUDIO_DIR $METRIC_DIR
