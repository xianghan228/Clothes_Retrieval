addpath(genpath('/home/lzhou/caffe-master/matlab'));
addpath('/home/lzhou/caffe-master/matlab/demo');
caffe.set_mode_cpu();
model_dir = '/home/lzhou/caffe-master/models/bvlc_reference_caffenet/';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
net = caffe.Net(net_model, net_weights, phase);
net.copy_from(net_weights);
net.layers('conv1').params(2).get_data()
[val,label] = net.blobs('prob').sort()
