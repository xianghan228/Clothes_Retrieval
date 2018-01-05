name  = load('/media/data_2/alibaba/kmeans/name/name0.mat');
fea = load('/media/data_2/alibaba/kmeans/vgg/For_XiaoLu/For_XiaoLu/0.mat');
name = (name.str_)';
size(name)
arr_ = fea.class0;
size(arr_);
opts = statset('Display','final');
tic;
[IDX,C,sumd,D] = kmeans(arr_,8,'dist','sqEuclidean','rep',1,'Options',opts);
save('/media/data_2/alibaba/kmeans/vgg/D_.mat','D');
cnt_every = 0;
IDX_last_real_1 = [];
IDX_last_real_2 = [];
for i = 1:8
    IDX_ = IDX - i;
    dis_ = D(:,i);
    for j = 1:length(IDX_)
        if(IDX_(j) == 0)
            cnt_every = cnt_every+1;
            dis_1(cnt_every) = dis_(j);
            IDX_1(cnt_every) = j;
            
        end
    end
    %cnt_every
    cnt_every = 0;
    [B,I] = sort(dis_1);
    len_ = length(dis_1);
    fprintf('%d',length(dis_1));
    for j = 1:len_
        IDX_last(j) = IDX_1(I(j));
    end
    %I1 = I(1:floor(0.7*len_))
    %I2 = (length(I1):len_)
    IDX_last_1 = IDX_last(1:floor(0.6*len_));
    IDX_last_2 = IDX_last((length(IDX_last_1)+1):len_);
    length(IDX_last_1);
    length(IDX_last_2);
    IDX_last_real_1 = [IDX_last_real_1 IDX_last_1];
    IDX_last_real_2 = [IDX_last_real_2 IDX_last_2];
     fprintf('%d',length(IDX_last_real_1))
     fprintf('%d  '  ,length(IDX_last_real_2))
    dis_1 = [];
    IDX_1 = [];
end
IDX_last_real_1;
IDX_last_real_2;
for i = 1:length(IDX_last_real_1)
    zero_one(IDX_last_real_1(i)) = 0;
end
for i = 1:length(IDX_last_real_2)
    zero_one(IDX_last_real_2(i)) = 1;
end
%zero_one = zero_one';
save('/media/data_2/alibaba/kmeans/vgg/zero_one.mat','zero_one');
in1 = find(IDX == 1);
in2 = find(IDX == 2);
in3 = find(IDX == 3);
save('/media/data_2/alibaba/kmeans/vgg/index.mat','IDX');
fprintf('%d ',size(in1))
fprintf('%d ',size(in2))
fprintf('%d ',size(in3))
num_dis1 = in1(1:5,:);
num_dis2 = in2(1:5,:);
num_dis3 = in3(1:5,:);
% size(num_dis,1)
% num_dis(2)
% name(3)
name_to_write1 = {};
name_to_write2 = {};
name_to_write3 = {};
for i = 1:size(num_dis1,1)
    name_to_write1{i} = name{num_dis1(i)};
end
for i = 1:size(num_dis2,1)
    name_to_write2{i} = name{num_dis2(i)};
end
for i = 1:size(num_dis3,1)
    name_to_write3{i} = name{num_dis3(i)};
end
% size(name_to_write,2)
for i = 1:size(name_to_write1,2)
    f1 = figure(1);
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write1{i}];
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
saveas(gcf,['/media/data_2/alibaba/pic_paper/','f1.jpg'])
hold on
for i = 1:size(name_to_write2,2)
    f2 = figure(2);
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write2{i}];
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
saveas(gcf,['/media/data_2/alibaba/pic_paper/','f2.jpg'])
hold on
for i = 1:size(name_to_write3,2)
    
    f3 = figure(3);
    path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write3{i}];
    if exist(path_,'file')
        fprintf('hello')
        f1 = imread(path_);
        subplot(5,5,i);
        imshow(f1);
    end
end
saveas(gcf,['/media/data_2/alibaba/pic_paper/','f3.jpg'])

hold on
