 name  = load('/media/data_2/alibaba/kmeans/name/name70.mat');
% fea = load('/media/data_2/alibaba/kmeans/vgg/For_XiaoLu/For_XiaoLu/276.mat');
 name = (name.str_)'
 size(name)
% arr_ = fea.class276;
% size(arr_)
% [IDX,C,sumd,D] = kmeans(arr_,6,'dist','sqEuclidean','rep',30);
index = load('/media/data_2/alibaba/kmeans/index/index70.mat');
IDX = index.IDX;
in1 = find(IDX == 1);
in2 = find(IDX == 2);
in3 = find(IDX == 3);
%save('/media/data_2/alibaba/kmeans/vgg/index.mat','IDX');
fprintf('%d ',size(in1))
fprintf('%d ',size(in2))
fprintf('%d ',size(in3))
num_dis1 = in1(1:15,:);
num_dis2 = in2(1:15,:);
num_dis3 = in3(1:15,:);
% size(num_dis,1)
% num_dis(2)
% name(3)
name_to_write1 = {};
name_to_write2 = {};
name_to_write3 = {};
for i = 1:size(num_dis1,1)
    name_to_write1{i} = name{num_dis1(i)}
end
for i = 1:size(num_dis2,1)
    name_to_write2{i} = name{num_dis2(i)}
end
for i = 1:size(num_dis3,1)
    name_to_write3{i} = name{num_dis3(i)}
end
% size(name_to_write,2)
for i = 1:size(name_to_write1,2)
    f1 = figure(1);
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write1{i}]
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
%saveas(gcf,['/media/data_2/alibaba/pic_paper/','f1.jpg'])
hold on
for i = 1:size(name_to_write2,2)
    f2 = figure(2);
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write2{i}]
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
%saveas(gcf,['/media/data_2/alibaba/pic_paper/','f2.jpg'])
hold on
for i = 1:size(name_to_write3,2)
    
    f3 = figure(3)
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write3{i}]
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
%saveas(gcf,['/media/data_2/alibaba/pic_paper/','f3.jpg'])

hold on
