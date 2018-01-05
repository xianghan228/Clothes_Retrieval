function [scores, maxlabel,a] = classification_vgg(im, use_gpu)
% [scores, maxlabel] = classification_demo(im, use_gpu)
%
% Image classification demo using BVLC CaffeNet.
%
% IMPORTANT: before you run this demo, you should download BVLC CaffeNet
% from Model Zoo (http://caffe.berkeleyvision.org/model_zoo.html)
%
% ****************************************************************************
% For detailed documentation and usage on Caffe's Matlab interface, please
% refer to Caffe Interface Tutorial at
% http://caffe.berkeleyvision.org/tutorial/interfaces.html#matlab
% ****************************************************************************
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%   maxlabel the label of the highest score
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = classification_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format in W x H x C with BGR channels:
%   % permute channels from RGB to BGR
%   im_data = im(:, :, [3, 2, 1]);
%   % flip width and height to make width the fastest dimension
%   im_data = permute(im_data, [2, 1, 3]);
%   % convert from uint8 to single
%   im_data = single(im_data);
%   % reshape to a fixed size (e.g., 227x227).
%   im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % subtract mean_data (already in W x H x C with BGR channels)
%   im_data = im_data - mean_data;

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
model_dir = '/media/data_2/alibaba/kmeans/vgg/';
net_model = [model_dir 'vgg_deploy.prototxt'];
net_weights = [model_dir 'vggs-finetune_iter_44000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

if nargin < 1
  % For demo purposes we will use the cat image
  fprintf('using caffe/examples/images/cat.jpg as input image\n');
  im = imread('../../examples/images/cat.jpg');
end

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im)};
toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
net.blobs('data').reshape([227 227 3 1])
scores = net.forward(input_data);
toc;

scores = scores{1};
scores = mean(scores, 2);  % take average scores over 10 crops
a = net.blobs('fc7').get_data()
%save('/media/data_2/alibaba/kmeans/vgg/fea.mat','a')
[~, maxlabel] = max(scores);

% call caffe.reset_all() to reset caffe
caffe.reset_all();

% ------------------------------------------------------------------------
function im_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
datar = load('/media/data_2/alibaba/siamese_models/datar.mat')
datag = load('/media/data_2/alibaba/siamese_models/datag.mat')
datab = load('/media/data_2/alibaba/siamese_models/datab.mat')

IMAGE_DIM = 227;
class(datar)
class(datar.datar)

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM],'nearest');  % resize im_data
%datar = permute(datar,[2,1]);
%datab = permute(datab,[2,1]);
%datag = permute(datag,[2,1]);
img_b = im_data(:,:,1)
im_data(:,:,1) = im_data(:,:,1) - (datar.datar);  % subtract mean_data (already in W x H x C, BGR)
im_data_b = im_data(:,:,1)
im_data(:,:,2) = im_data(:,:,2) - (datag.datag);
img_g = im_data(:,:,2)
im_data(:,:,3) = im_data(:,:,3) - (datab.datab);



