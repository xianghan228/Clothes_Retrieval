#!/usr/bin/env sh
#/data/pxliang/workspace/caffe-master-tripletilzh/build/tools/extract_features ./../models_wtbuy/vggnet_clothes_woman_triplet1024_iter_150000.caffemodel ./deploy_softmax_mfq.prototxt fc6 JDNanzhuang 17175 lmdb GPU 6 > meta_features_JD.txt
if [ $# ne 6 ];then
   echo "$0 caffemodel deploy layer savelmdb datasize gpu"
   exit
fi
caffemodel=$1
deploynet=$2
layer=$3
savepath=$4
datasize=$5
gpu=$6

/data/pxliang/pzz/caffe-master/build/tools/extract_retrieval_features ${caffemodel} ${deploynet} ${layer} ${savepath} ${datasize} lmdb GPU ${gpu} > meta_features_JD_triplet500_Nv.txt
