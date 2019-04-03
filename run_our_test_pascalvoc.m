%setup;
clear all;
close all; clc;

models_vec=[60];
%models_vec=[25];
VOCinit;
fd = '/home/mlcv/pami_experiments/VOCdevkit/VOC2007/JPEGImages/';

for it=12:20,
    cls = VOCopts.classes{it};
    VOCopts.testset = 'test';
    [ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'test'),'%s %d');
    
    for jj=1:length(models_vec),
        
        smodel=int2str(models_vec(jj));
        
        % loading cifar model
        net_path = ['/home/mlcv/pami_experiments/VOCdevkit/data/exp/pascalvoc2007_normal_' cls '-resnet-101/net-epoch-' smodel '.mat'];
        %net_path2 = '/home/mlcv/Desktop/ResNet-Matconvnet/data/models/imagenet-resnet-101-dag.mat';
        
        net_our = load(net_path)
        % net_imagenet = load(net_path2)
        
        net=dagnn.DagNN.loadobj(net_our.net);
        net.mode = 'test' ;
        net.accumulateParamDers = 0;
        net.conserveMemory = 1;
        net.parameterServer = [];
        
        net.vars(end-3).precious = 1;
        net.move('gpu');
        %gpuDevice(1)
        count = 0;
        estimated_labels=[];
        
        batchsize = 20;
        addpath([cd '/VOCcode']);
        
        ss = 0;
        %net.mode = 'normal';
        labels=classifier.gt;
        net.mode = 'test' ;
        
        load(['/home/mlcv/pami_experiments/VOCdevkit/data/exp/pascalvoc2007_normal_' cls '-resnet-101/imageStats.mat']);
        opts.numThreads = 12;
        opts.imageSize = [224 224 3];
        opts.border = [32 32];
        opts.averageImage = rgbMean;
        %opts.rgbVariance = rgbCovariance;
        opts.transformation = 'stretch';
        opts.numAugments = 1;
        
        
        for i=1:248%length(ids),
            if i==248,
                batchsize = 12;
            end
            i
            %imtemp=zeros(224,224,3,batchsize);
            % ss=0;
            clear imt
            for j=1:batchsize,
                ss = ss + 1;
                imname = ids(ss);
                imfile = [fd imname{1} '.jpg'];
                imt{j} = imfile;
            end
            
            imtemp = cnn_imagenet_get_batch(imt, opts, ...
                'prefetch', false, ...
                'transformation', 'none') ;
            count=count+batchsize;
            
            input = {'data', gpuArray(single(imtemp)), 'label', single(labels(count-batchsize+1:count)')};
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
        RR(jj)=info.ap;
        
        
        
        fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');
        for i=1:length(ids),
            % write to results file
            c=estimated_labels(i);
            fprintf(fid,'%s %f\n',ids{i},c);
        end
        
        [recall,prec,ap]=VOCevalcls(VOCopts,'comp1',cls,true);
        RRVOC(jj)=ap;
    end
    RRtot{it}=RR;
    RRVOCtot{it}=RRVOC;
end

