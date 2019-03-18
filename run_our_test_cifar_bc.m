%setup;
clear all;
close all; clc;
% loading cifar model
load('test_batch.mat')
net_path = '/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/cifar10-resnet-101/net-epoch-18.mat';
%net_path2 = '/home/mlcv/Desktop/ResNet-Matconvnet/data/models/imagenet-resnet-101-dag.mat';

net_our = load(net_path)
% net_imagenet = load(net_path2)

net=dagnn.DagNN.loadobj(net_our.net);

labels = double(labels+1);
batchsize = 20;
fd = ['/home/mlcv/Desktop/ResNet-Matconvnet/dataset/cifar10/cifar_test_images/'];
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
for i=1:500,
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
        load im_parameters
        load('/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/cifar10-resnet-101/imageStats.mat')
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
jj=find(labels~=1);
labels_new=labels;
labels_new(jj)=-1;
[RECALL, PRECISION, info] = vl_pr(labels_new, estimated_labels);
fprintf('\n Task is accomplished');        
