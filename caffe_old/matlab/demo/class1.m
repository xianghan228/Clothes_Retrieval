%im = imread('/home/lzhou/caffe-2015-siamese//examples/images/cat.jpg')
im = imread('/media/data_2/alibaba/pic/001330127301.jpg')
[scores, maxlabel,transformed_im] = classification_demo(im, 0);
%transformed_image_py = load('/media/data_2/alibaba/image1_process.mat');
%transformed_image_py_real = permute(transformed_image_py.transformed_image,[2,3,1]);