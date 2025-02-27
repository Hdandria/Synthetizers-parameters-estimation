#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=6G
#$ -pe smp 1
#$ -l centos
#$ -l node_type=ddy
#$ -cwd
#$ -j y
#$ -o dlogs/
#$ -e dlogs/
#$ -t 1-8

JOB_PARAMS=$(sed -n "${SGE_TASK_ID}p" jobs/render/render-jobs.txt)
JOB_DIR=$(echo $JOB_PARAMS | cut -d' ' -f1)
DATASET=$(echo $JOB_PARAMS | cut -d' ' -f2)

PRED_DIR=${JOB_DIR}/predictions
AUDIO_DIR=${JOB_DIR}/audio

mkdir -p $AUDIO_DIR

module load hdf5-parallel
apptainer exec ../containers/torch.sif ./renderscript.sh $PRED_DIR $AUDIO_DIR $DATASET
