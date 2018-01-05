for i = 200:439
    name  = load(sprintf('/media/data_2/alibaba/kmeans/name/name%d.mat',i-1));
    fea = load(sprintf('/media/data_2/alibaba/kmeans/vgg/For_XiaoLu/For_XiaoLu/%d.mat',i-1));
    name = (name.str_)';
    size(name);
    str_class = sprintf('class%d',i-1);
    arr_ = eval(['fea.', num2str(str_class)]);
    every_class_num = size(name,1);
    if every_class_num <=50
        num_class = 1;
    end
    if  every_class_num <1000 && every_class_num >50
        num_class = ceil(every_class_num/50);
    end
    if every_class_num >=1000 && every_class_num <2000
        num_class = 12;
    end
    if  every_class_num <5000 && every_class_num >=2000
        num_class = 8
    end
    if every_class_num >=5000
        num_class = 5
    end
    if every_class_num<1000 
        iter_num = 15;
    end
    if every_class_num>=1000 
        iter_num = 30;
    end
    [IDX,C,sumd,D] = kmeans(arr_,num_class,'dist','sqEuclidean','rep',iter_num);
    fprintf('iter_num%d,num_class%d,',iter_num,num_class)
    save(sprintf('/media/data_2/alibaba/kmeans/index/index%d.mat',i-1),'IDX');
    fprintf('hello%d ',i)
    
    
    
    
    
    
    
    
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