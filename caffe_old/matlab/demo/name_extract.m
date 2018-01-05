f_read = fopen('/media/data_2/alibaba/kmeans/final_train_label_list_reduce_perfect.txt','r');
cnt = 0;
str_ = {};
num_his = 0
while ~feof(f_read)
    
    cnt = cnt + 1;
    line1 = fgetl(f_read);
    num = str2num(line1(18:end));
% %     if(num~=0)
% % %         break;
% % %     end;
    if num ~= num_his
        path_save = sprintf('/media/data_2/alibaba/kmeans/name/name%d.mat',num_his);
        save(path_save,'str_');
        str_ = {};
        cnt = 1
    end
    img_name = line1(1:16);
    fprintf('%s\n',img_name);
    str_{cnt} = img_name;
    fprintf('%d\n',num);
    
    
    num_his = num;
end

path_save = sprintf('/media/data_2/alibaba/kmeans/name/name%d.mat',num_his);
save(path_save,'str_');
%fprintf('%d',size(str_,2));
%arr_len = size(str_,2)
%save('/media/data_2/alibaba/kmeans/vgg/name1.mat','str_');