#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/l2normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype normsqr;
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  caffe_gpu_powx(n*d, bottom_data, Dtype(2), squared_data);
  for (int i=0; i<n; ++i) {
    caffe_gpu_asum<Dtype>(d, squared_data+i*d, &normsqr);
    caffe_gpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);
  }
   const Dtype* bottom_data1 = bottom[0]->cpu_data();
   Dtype* top_data1 = top[0]->mutable_cpu_data();

  //LOG(INFO)<<"L2NORM_input: "<<bottom_data1[0]<<" "<<bottom_data1[1]<<" "<<bottom_data1[2];
  //LOG(INFO)<<"L2NORM_input: "<<top_data1[0]<<" "<<top_data1[1]<<" "<<top_data1[2];
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->num();
  int d = top[0]->count() / n;
  Dtype a;
  for (int i=0; i<n; ++i) {
    caffe_gpu_dot(d, top_data+i*d, top_diff+i*d, &a);
    caffe_gpu_scale(d, a, top_data+i*d, bottom_diff+i*d);
    caffe_gpu_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
    caffe_gpu_dot(d, bottom_data+i*d, bottom_data+i*d, &a);
    caffe_gpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
  }
}

// INSTANTIATE_CLASS(L2NormalizeLayer);
INSTANTIATE_LAYER_GPU_FUNCS(L2NormalizeLayer);

}  // namespace caffe
