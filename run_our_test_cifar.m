%setup;
clear all;
close all; clc;
% loading cifar model
load('test_batch.mat')

net_path = '/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/cifar_multi_class_results/net-epoch-20.mat';
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
labels_new=labels,
for i=2:9,
    jj=find(labels==i);
    labels_new(jj)=i+1;
end
ll=find(labels==10);
labels_new(ll)=2;
labels=labels_new;

load im_parameters
load('/home/mlcv/Desktop/ResNet-Matconvnet/data/exp/cifar_multi_class_results/imageStats.mat')
for i=1:500,
    i
    imtemp=zeros(224,224,3,batchsize);
    %ss=0;
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
        %    if ~isempty(opts.rgbVariance)
         %       offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
          %  end
            imtemp(:,:,:,j) = bsxfun(@minus, single(imt(sx,sy,:)), offset) ;
            
        end
       % imtemp(:,:,:,j) = single(imt(sx,sy,:));

    end
    count=count+batchsize;
    
    input = {'data', gpuArray(single(imtemp)), 'label', labels(count-batchsize+1:count)};
    net.eval(input);
    output = net.vars(end-3).value;
    scores = squeeze (output);
    [gg ll] = max(scores);
    [ll; labels(count-batchsize+1:count)']
    estimated_labels=[estimated_labels ll];

end

gg=find((estimated_labels-labels')==0);
RR=length(gg)/length(labels)
fprintf('\n Task is accomplished');        
