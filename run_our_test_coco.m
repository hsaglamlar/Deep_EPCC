%setup;
clear all;
close all; clc;


%models_vec=[25];
fd = '/home/mlcv/ResNet-Matconvnet/dataset/COCO/';
gpuDevice(3)
% loading cifar model
net_path = ['/home/mlcv/ResNet-Matconvnet/data/exp/coco_bs128-resnet-101/net-epoch-98.mat'];
%net_path = ['/home/mlcv/ResNet-Matconvnet/net-epoch-80.mat'];

net_our = load(net_path)
% net_imagenet = load(net_path2)

net=dagnn.DagNN.loadobj(net_our.net);
net.mode = 'test' ;
net.accumulateParamDers = 0;
net.conserveMemory = 1;
net.parameterServer = [];

net.vars(end-3).precious = 1;
net.move('gpu');

count = 0;
estimated_labels=[];

batchsize = 20;

ss = 0;
%net.mode = 'normal';
net.mode = 'test' ;

load(['/home/mlcv/ResNet-Matconvnet/data/exp/coco_bs128-resnet-101/imageStats.mat']);
opts.numThreads = 12;
opts.imageSize = [224 224 3];
opts.border = [32 32];
opts.averageImage = rgbMean;
%opts.rgbVariance = rgbCovariance;
opts.transformation = 'stretch';
opts.numAugments = 1;

%fid = fopen('/home/mlcv/ResNet-Matconvnet/dataset/COCO/coco/datasets/coco/coco_test_imglist.txt','r');
images = (importdata('/home/mlcv/ResNet-Matconvnet/dataset/COCO/coco/datasets/coco/coco_test_imglist.txt'));
labels = (importdata('./datasets/coco/coco_test_label.txt')==1);
estimated_labels = zeros(40504,80);
ss=0;
for i=1:2026%length(ids),
    if i==2026,
        batchsize = 4;
    end
    i
    %imtemp=zeros(224,224,3,batchsize);
    % ss=0;
    clear imt
    for j=1:batchsize,
        %tline = fgetl(fid);
        ss = ss +1 ;
        tline = images.textdata{ss};
        imfile = [fd tline];
        imt{j} = imfile;
    end
    
    imtemp = cnn_imagenet_get_batch(imt, opts, ...
        'prefetch', false, ...
        'transformation', 'none') ;
    count=count+batchsize;
    
   % input = {'data', gpuArray(single(imtemp)), 'label', single(labels(count-batchsize+1:count)')};
     input = {'data', gpuArray(single(imtemp))};
    
    net.eval(input);
    
    output = net.vars(end-3).value;
    scores = squeeze (output);
    %[gg ll] = max(scores);
    estimated_labels ((count-batchsize+1:count),:)=gather(scores');
    
end

save coco_results estimated_labels

