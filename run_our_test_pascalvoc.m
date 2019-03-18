%setup;
clear all;
close all; clc;
% loading cifar model
net_path = '/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/pascalvoc2007-resnet-101/net-epoch-17.mat';
%net_path2 = '/home/mlcv/Desktop/ResNet-Matconvnet/data/models/imagenet-resnet-101-dag.mat';

net_our = load(net_path)
% net_imagenet = load(net_path2)

net=dagnn.DagNN.loadobj(net_our.net);
batchsize = 1;
addpath([cd '/VOCcode']);
VOCinit;
cls=VOCopts.classes{1};
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'test'),'%s %d');
fd = '/home/mlcv/yedekdisk/pami_experiments/VOCdevkit/VOC2007/JPEGImages';
ss = 0;
%net.mode = 'normal';
net.mode = 'test' ;
net.accumulateParamDers = 0;
net.conserveMemory = 1;
net.parameterServer = [];

net.vars(end-3).precious = 1;
net.move('gpu');
count = 0;
estimated_labels=[];
load im_parameters
load('/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/pascalvoc2007-resnet-101/imageStats.mat');,
labels=classifier.gt;
for i=1:length(ids),
    i
    imtemp=zeros(224,224,3,batchsize);
   % ss=0;
    for j=1:batchsize,
        ss = ss + 1;
        imname = int2str(ss);
        imfile = [fd imname '.jpg'];
        imt = imread(imfile);
        w = size(imt,2) ;
        h = size(imt,1) ;
        
        opts.averageImage=averageImage;
        opts.rgbVariance=rgbCovariance;
        if ~isempty(opts.averageImage)
            offset = opts.averageImage ;
           % if ~isempty(opts.rgbVariance)
            %    offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
            %end
            %offset = reshape(offset,1,1,3);
            imtemp(:,:,:,j) = bsxfun(@minus, single(imt(sx,sy,:)), offset) ;
            
        end
        %imtemp(:,:,:,j) = single(imt(sx,sy,:));

    end
    count=count+batchsize;
    
    input = {'data', gpuArray(single(imtemp)), 'label', labels(count-batchsize+1:count)};
   % input = {'data', gpuArray(single(imtemp))};

    net.eval(input);
    
    output = net.vars(end-3).value;
    scores = squeeze (output);
    %[gg ll] = max(scores);
    estimated_labels=[estimated_labels; scores];

end
labels_new=classifier.gt;
[RECALL, PRECISION, info] = vl_pr(labels_new, estimated_labels);
fprintf('\n Task is accomplished');        
