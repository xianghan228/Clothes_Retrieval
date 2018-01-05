#!/bin/bash

TOOLS=/data1/qtang/samsung/caffe/build/tools/

LOG=".log.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")


$TOOLS/caffe train --gpu=7 --solver=./solver.prototxt  --weights=/data1/zwshen/alibaba/train_MM_2/pool6_siamese_train/models.caffemodel 
