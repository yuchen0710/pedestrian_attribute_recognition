#!/usr/bin/env python

python ./script/experiment/train_deepmar_resnet50.py \
    --sys_device_ids="(0,)" \
    --dataset=peta \
    --test_split=test \
    --resize="(224,224)" \
    --exp_subpath=deepmar_resnet50 \
    --run=1 \
    --test_only=True \
    --model_weight_file='./exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch1.pth'