clear all;
close all; clc;

labels_temp = double(importdata('/home/mlcv/ResNet-Matconvnet/dataset/COCO/coco/datasets/coco/coco_train_label.txt')==1);  
hh=sum(labels_temp');
empty_labels=find(hh==0);
count = 0;
for i=1:length(hh),
    if ismember(i,empty_labels);
        continue;
    else
        count = count + 1
        labels (count,:) = labels_temp(i,:);
    end
end

images = (importdata('/home/mlcv/ResNet-Matconvnet/dataset/COCO/coco/datasets/coco/coco_train_imglist.txt')); 

% ************************* Writing IMDB ************************
% ---------------------------------------------------------------
datasetDir = '/home/mlcv/ResNet-Matconvnet/dataset/COCO/train2014';

opts.seed = 0 ;             % random seed generator
opts.ratio = [0.98 0.02];     % train:val ratio
opts.ext = '.jpg';          % extension of target files
opts.per_class_limit = inf; % inf indicates no limit

opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2));
rng(opts.seed);
imdb.imageDir = datasetDir;

% meta
imdb.meta.classes = [1:80];
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', numel(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.label   = [];
imdb.images.set     = []; 
imdb.images.label  =zeros(1,82081,80);
count = 0;
for i=1:length(hh),
    i
    if ismember(i,empty_labels);
        continue;
    else
        count = count + 1;
        imname = images.textdata{i};
        jj = strfind(imname,'/');
        imname(jj+1:end)
        imdb.images.name {count} =  imname(jj+1:end);
        imdb.images.label (1,count,:) = labels(count,:);
        %curr_set = ones(1,floor(opts.ratio(1)*numel(curr_name)));
        %curr_set = [curr_set 2*ones(1,numel(curr_name)-numel(curr_set))];
        imdb.images.set (1,count)= 1;
        %fprintf(' done!\n');
    end
end

% id
imdb.images.id = 1:length(imdb.images.name);

save('coco_imdb.mat', 'imdb');

