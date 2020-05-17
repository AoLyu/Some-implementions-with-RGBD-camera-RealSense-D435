#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=2

python receiver.py --model eval_RT/pose_model.pth\
 --refine_model eval_RT/pose_refine_model.pth\
 --resume_model eval_RT/seg_model.pth
