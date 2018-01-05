 #include <algorithm> 
 #include <cmath> 
 #include <cfloat> 
 #include <vector> 
  
 #include "caffe/layer.hpp"
 #include "caffe/loss_layers.hpp" 
 #include "caffe/vision_layers.hpp" 
 #include "caffe/util/math_functions.hpp" 
 #include "caffe/util/io.hpp" 
 #include "omp.h"
 namespace caffe { 
 
 using std::max; 
  
 template <typename Dtype> 
 void TripletRankingHingeLossLayer<Dtype>::Forward_cpu( 
     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
     const Dtype* query_data = bottom[0]->cpu_data(); 
     const Dtype* similar_sample_data = bottom[1]->cpu_data(); 
     const Dtype* dissimilar_sample_data = bottom[2]->cpu_data();

   Dtype* query_diff = bottom[0]->mutable_cpu_diff(); 
   Dtype* similar_sample_diff = bottom[1]->mutable_cpu_diff(); 
   Dtype* dissimilar_sample_diff = bottom[2]->mutable_cpu_diff(); 
   Dtype  margin=this->layer_param_.triplet_ranking_hinge_loss_param().margin();

   
   int num = bottom[0]->num(); 
   int count = bottom[0]->count(); 
   int dim = count / num;

   caffe_sub(count, query_data, similar_sample_data,similar_sample_diff); 
   caffe_sub(count, query_data, dissimilar_sample_data,dissimilar_sample_diff);
   caffe_sub(count, similar_sample_data, dissimilar_sample_data, query_diff);
  
   Dtype* loss = top[0]->mutable_cpu_data();
   
   Dtype query_similar_distance_norm; 
   Dtype query_dissimilar_distance_norm; 
   switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) { 
   case TripletRankingHingeLossParameter_Norm_L1: { 
     for (int i = 0; i < num; ++i) { 
       query_similar_distance_norm = caffe_cpu_asum( 
          dim, similar_sample_diff + bottom[1]->offset(i)); 
       query_dissimilar_distance_norm = caffe_cpu_asum( 
           dim, dissimilar_sample_diff + bottom[2]->offset(i)); 
       loss[0] += max(Dtype(0), query_similar_distance_norm - 
                  query_dissimilar_distance_norm + margin);
       
     } 
     break; 
   } 
   case TripletRankingHingeLossParameter_Norm_L2: { 
     for (int i = 0; i < num; ++i) { 
       query_similar_distance_norm = caffe_cpu_dot( 
          dim, similar_sample_diff + bottom[1]->offset(i), 
           similar_sample_diff + bottom[1]->offset(i)); 
       query_dissimilar_distance_norm = caffe_cpu_dot( 
           dim, dissimilar_sample_diff + bottom[2]->offset(i), 
          dissimilar_sample_diff + bottom[2]->offset(i));
        
       loss[0] += max(Dtype(0), query_similar_distance_norm -  query_dissimilar_distance_norm + margin);
     
    } 
     break; 
  } 
   default: { 
    LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " << 
         this->layer_param_.triplet_ranking_hinge_loss_param().norm(); 
   } 
   } 
   loss[0] =loss[0] / num;
 } 
  
 template <typename Dtype> 
 void TripletRankingHingeLossLayer<Dtype>::Backward_cpu( 
     const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
     const vector<Blob<Dtype>*>& bottom) { 
    Dtype  margin=this->layer_param_.triplet_ranking_hinge_loss_param().margin();
   
   if (propagate_down[0]) { 
    Dtype* query_diff = bottom[0]->mutable_cpu_diff(); 
    Dtype* similar_sample_diff = bottom[1]->mutable_cpu_diff(); 
    Dtype* dissimilar_sample_diff = bottom[2]->mutable_cpu_diff(); 
     int num = bottom[0]->num(); 
     int count = bottom[0]->count();
     int dim= count/num;
 
   Dtype query_similar_distance_norm; 
   Dtype query_dissimilar_distance_norm; 
   Dtype loss ;
   
     switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) { 
     case TripletRankingHingeLossParameter_Norm_L1: { 
       caffe_cpu_sign(count, similar_sample_diff, similar_sample_diff); 
      caffe_scal(count, Dtype(-1. / num), similar_sample_diff); 
       caffe_cpu_sign(count, dissimilar_sample_diff, dissimilar_sample_diff); 
       caffe_scal(count, Dtype(1. / num), dissimilar_sample_diff); 
       break; 
    } 
     case TripletRankingHingeLossParameter_Norm_L2: {
      
       for (int i = 0; i < num; ++i) {  
      query_similar_distance_norm = caffe_cpu_dot(  
          dim, similar_sample_diff + bottom[1]->offset(i),  
          similar_sample_diff + bottom[1]->offset(i));  
      query_dissimilar_distance_norm = caffe_cpu_dot(  
          dim, dissimilar_sample_diff + bottom[2]->offset(i),  
          dissimilar_sample_diff + bottom[2]->offset(i));  
      loss= query_similar_distance_norm - query_dissimilar_distance_norm + margin; 
     if (loss>0){
     caffe_scal(dim, Dtype(-2. / num), query_diff+bottom[0]->offset(i));
     caffe_scal(dim, Dtype(-2. / num), similar_sample_diff+bottom[1]->offset(i));  
     caffe_scal(dim, Dtype(2. / num), dissimilar_sample_diff+bottom[2]->offset(i));
        }
     else {
     caffe_scal(dim, Dtype(0.), query_diff+bottom[0]->offset(i));
     caffe_scal(dim, Dtype(0.), similar_sample_diff+bottom[1]->offset(i));  
     caffe_scal(dim, Dtype(0.), dissimilar_sample_diff+bottom[2]->offset(i));
        }
     Dtype* query_ind=query_diff+bottom[0]->offset(i);
     Dtype* simi_ind=similar_sample_diff+bottom[1]->offset(i);
     Dtype* dissimi_ind=dissimilar_sample_diff+bottom[2]->offset(i);
     LOG(INFO)<<"query_diff: "<< query_ind[1] << query_ind[100] <<query_ind[1000];
     LOG(INFO)<<"simi_diff: "<< simi_ind[1] << simi_ind[100] <<simi_ind[1000];
     LOG(INFO)<<"dissimi_diff: "<< dissimi_ind[1] << dissimi_ind[100] <<dissimi_ind[1000];
     }               
                            
       break; 
     } //case
     default: { 
       LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " << 
           this->layer_param_.triplet_ranking_hinge_loss_param().norm(); 
              } 
     }// switch
    
  
   }//if (propagate_down[0]) 
 }//void TripletRankingHingeLossLayer<Dtype>::Backward_cpu 
  
 INSTANTIATE_CLASS(TripletRankingHingeLossLayer); 
 REGISTER_LAYER_CLASS(TripletRankingHingeLoss); 
 }  // namespace caffe 

