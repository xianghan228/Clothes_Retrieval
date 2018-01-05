im = imread('/home/lzhou/caffe-2015-siamese/examples/images/cat.jpg');
[scores, maxlabel] = classification_demo1(im, 0);
save_dir = '/media/data_2/alibaba/siamese_models/feature_save/';

a = dir(save_dir);
b= a.bytes