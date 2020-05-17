#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=2

python receiver.py --resume_model model_190_0.002232187078159768.pth
