% ------------------------------------------------------------------------
function images = prepare_batch(image_files,IMAGE_MEAN,batch_size,IMAGE_DIM,img_dir)
% ------------------------------------------------------------------------

num_images = length(image_files);
if nargin < 3
    batch_size = num_images;
end

Res_dim=IMAGE_DIM;

% num_images = length(image_files);
images = zeros(Res_dim,Res_dim,3,batch_size,'single');

parfor i=1:num_images
%     % read file
%     fprintf('%c Preparing %s\n',13,image_files{i});
    try
        im = imread(strcat(img_dir,image_files{i}));
        im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
        im_data = permute(im_data, [2, 1, 3]);  % flip width and height
        im_data = single(im_data);  % convert from uint8 to single
        im_data = imresize(im_data, [Res_dim Res_dim], 'bilinear');  % resize im_data
        im_data = im_data - IMAGE_MEAN; 
        images(:,:,:,i)=im_data;
    catch
        warning('Problems with file',image_files{i});
    end
end
