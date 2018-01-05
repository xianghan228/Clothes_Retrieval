%feature = ones(2,4096)
%im = imread('/media/data_2/alibaba/pic2/201150000066.jpg');
% [scores, maxlabel,fea] = classification_vgg(im, 0);
% feature(1,:) = fea;
% im = imread('/media/data_2/alibaba/pic2/201150000066.jpg');
% [scores, maxlabel,fea] = classification_vgg(im, 0);
% feature(2,:) = fea;
% save('/media/data_2/alibaba/kmeans/vgg/fea.mat','feature')
f_read = fopen('/media/data_2/alibaba/kmeans/final_train_label_list_reduce.txt','r');
cnt = 0;
str_ = {};
while ~feof(f_read)
    cnt = cnt + 1;
    line1 = fgetl(f_read);
    num = str2num(line1(18:end));
    img_name = line1(1:16);
    fprintf('%s\n',img_name);
    str_{cnt} = img_name;
    fprintf('%d\n',num);
    if(num~=0)
        break;
    end;
    
    num_his = num;
end
fprintf('%d',size(str_,2));
arr_len = size(str_,2)
save('/media/data_2/alibaba/kmeans/vgg/name.mat','str_');
feature = ones(arr_len,4096)
for i = 1:arr_len
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' str_{i}];
    if exist(path_,'file')
        fprintf('hello');
        im = imread(path_);
        [scores, maxlabel,fea] = classification_vgg(im, 0);
        feature(i,:) = fea;
    else
        feature(i,:) = ones(1,4096);
        fprintf('error');
    end
    
end
save('/media/data_2/alibaba/kmeans/vgg/fea.mat','feature');