for i = 27:50
    name  = load(sprintf('/media/data_2/alibaba/kmeans/name/name%d.mat',i-1));
    fea = load(sprintf('/media/data_2/alibaba/kmeans/vgg/For_XiaoLu/For_XiaoLu/%d.mat',i-1));
    name = (name.str_)';
    size(name);
    str_class = sprintf('class%d',i-1);
    arr_ = eval(['fea.', num2str(str_class)]);
    every_class_num = size(name,1);
    num_class = ceil(every_class_num/100);
    
    
    [IDX,C,sumd,D] = kmeans(arr_,num_class,'dist','sqEuclidean','rep',1);
    
    zero_one = [];
    cnt_every = 0;
    IDX_last_real_1 = [];
    IDX_last_real_2 = [];
    for m = 1:num_class
        IDX_ = IDX - m;
        dis_ = D(:,m);
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
        %fprintf('%d',length(dis_1));
        for j = 1:len_
            IDX_last(j) = IDX_1(I(j));
        end
        %I1 = I(1:floor(0.7*len_))
        %I2 = (length(I1):len_)
        IDX_last_1 = IDX_last(1:floor(0.7*len_));
        IDX_last_2 = IDX_last((length(IDX_last_1)+1):len_);
        length(IDX_last_1);
        length(IDX_last_2);
        IDX_last_real_1 = [IDX_last_real_1 IDX_last_1];
        IDX_last_real_2 = [IDX_last_real_2 IDX_last_2];
        %fprintf('%d',length(IDX_last_real_1))
        %fprintf('%d  '  ,length(IDX_last_real_2))
        dis_1 = [];
        IDX_1 = [];
    end
    IDX_last_real_1;
    IDX_last_real_2;
    for m = 1:length(IDX_last_real_1)
        zero_one(IDX_last_real_1(m)) = 0;
    end
    for m = 1:length(IDX_last_real_2)
        zero_one(IDX_last_real_2(m)) = 1;
    end
    %zero_one = zero_one';
    save(sprintf('/media/data_2/alibaba/kmeans/z_o/zero_one%d.mat',i-1),'zero_one');
    fprintf('num_class%d,',num_class);
    save(sprintf('/media/data_2/alibaba/kmeans/index/index%d.mat',i-1),'IDX');
    fprintf('hello%d ',i);
    
    
    
    
    
    
    
    
%     in1 = find(IDX == 1);
%     in2 = find(IDX == 2);
%     in3 = find(IDX == 3);
%     num_dis1 = in1(1:5,:);
%     num_dis2 = in2(1:5,:);
%     num_dis3 = in3(1:5,:);
%     
%     name_to_write1 = {};
%     name_to_write2 = {};
%     name_to_write3 = {};
%     for j = 1:size(num_dis1,1)
%         name_to_write1{j} = name{num_dis1(j)}
%     end
%     for j = 1:size(num_dis2,1)
%         name_to_write2{j} = name{num_dis2(j)}
%     end
%     for j = 1:size(num_dis3,1)
%         name_to_write3{j} = name{num_dis3(j)}
%     end
%     for j = 1:size(name_to_write1,2)
%         f1 = figure(1);
%         path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write1{j}]
%         if exist(path_,'file')
%             fprintf('hello')
%             f1 = imread(path_);
%             subplot(5,5,j);
%             imshow(f1);
%         end
%     end
%     saveas(gcf,['/media/data_2/alibaba/pic_paper/','f1.jpg'])
%     hold on
%     for j = 1:size(name_to_write2,2)
%         f2 = figure(2);
%         path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write2{j}]
%         if exist(path_,'file')
%             fprintf('hello')
%             f1 = imread(path_);
%             subplot(5,5,j);
%             imshow(f1);
%         end
%     end
%     
%     hold on
%     for j = 1:size(name_to_write3,2)
%     
%         f3 = figure(3)
%         path_ = ['/media/data_2/alibaba/pic2/home/public/zwfang/final_train_227_227_matlab/' name_to_write3{j}]
%         if exist(path_,'file')
%             fprintf('hello')
%             f1 = imread(path_);
%             subplot(5,5,j);
%             imshow(f1);
%         end
%     end
% 
% 
%     hold on
% 
end