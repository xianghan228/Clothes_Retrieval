addpath(genpath('/home/lzhou/caffe-2015-siamese/matlab'));
addpath('/home/lzhou/caffe-2015-siamese/matlab/demo');
caffe.set_mode_cpu();
model_dir = '/media/data_2/alibaba/siamese_models/';
net_model = [model_dir 'ali_siamese_trainval_deploy.prototxt'];
net_weights = [model_dir 'models.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
net = caffe.Net(net_model, net_weights, phase);
net.copy_from(net_weights);
%net.layers('conv1').params(2).get_data()
%net.layers('feat').params(1).get_data()
%net.blobs('data').get_data()


net.layers('L2Norm_fc7')