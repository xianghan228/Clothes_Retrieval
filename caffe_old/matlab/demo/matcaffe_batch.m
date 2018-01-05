function [scores] = matcaffe_batch(list_im, use_gpu,dim,IMAGE_DIM,img_dir)
% If you have multiple images, cat them with cat(4, ...)

% Add caffe/matlab to you Matlab search PATH to use matcaffe
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.

 net_model='VGG_ILSVRC_19_layers_deploy.prototxt';
 %net_model = 'zeiler_deploy.prototxt';
 %net_weights = '/home/iva/Documents/Caffe.Installation/caffe-video_triplet-master-xlwang/rank_scripts/rank_zeiler/model/vgg19-fn_iter_10000.caffemodel';
 net_weights = '/home/iva/vgg19_muti-loss-1-2_iter_6901.caffemodel';
 phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please find a model before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

if nargin < 1
  % For test purposes
  list_im = {'peppers.png','onions.png'};
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% prepare oversampled input
% input_data is Height x Width x Channel x Num
batch_size=10;
if mod(length(list_im),batch_size)
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end
%load mean data
mean_vec=[103.939,116.779,123.68];%vgg16=[104.00699,116.66877,122.67892]
Image_mean=zeros(IMAGE_DIM,IMAGE_DIM,3,'single');
Image_mean(:,:,1)=mean_vec(1)*ones(IMAGE_DIM,IMAGE_DIM);
Image_mean(:,:,2)=mean_vec(2)*ones(IMAGE_DIM,IMAGE_DIM);
Image_mean(:,:,3)=mean_vec(3)*ones(IMAGE_DIM,IMAGE_DIM);

num_images = length(list_im);
scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size);
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = prepare_batch(list_im(range),Image_mean,batch_size,IMAGE_DIM,img_dir);
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = net.forward({input_data});
    toc
    %output_data = squeeze(output_data{1});
    output_data=output_data{1};
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
%     output_data_sum = squeeze(sum(sum(output_data{1})));
    % output_data_max = squeeze(max(max(output_data{1})));
%     scores(1:dim,range) = output_data_sum(:,mod(range-1,batch_size)+1);
     %scores(1:dim,range) = output_data_max(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);

% call caffe.reset_all() to reset caffe
caffe.reset_all();

% ------------------------------------------------------------------------
