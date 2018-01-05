#/bin/bash

TOOLS=/data1/qtang/samsung/caffe/build/tools

LOG=".log.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")


$TOOLS/caffe train --gpu=2 --solver=./solver.prototxt  --weights=./part_train_withfc7_.caffemodel
