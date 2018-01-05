fea  = load('/media/data_2/alibaba/kmeans/vgg/0.mat');
name = load('/media/data_2/alibaba/kmeans/vgg/name1.mat');
name = (name.str_)'
size(name)
arr_ = fea.class0;
no_dims = 2;
initial_dims = 50;
perplexity = 30;
mappedX = tsne(arr_, [], no_dims, initial_dims, perplexity);
gscatter(mappedX(:,1), mappedX(:,2));
