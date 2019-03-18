% run_our_experiments([101], 'gpus', [1]);
function run_our_experiments_pascalvoc(Ns,imdb_name,cls, varargin)
% Usage example: run_experiments([18 34 50 101 152], 'gpus', [1 2 3 4]); 
% On you own dataset: run_experiments([18 34 50 101],'datasetName',...
% 'reflectance', 'datafn', @setup_imdb_reflectance, 'gpus', [1 2]);
% Options: 
%   'expDir'['exp'], 'bn'[true], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   'meanType'['image'], 'whitenData'[true], 'contrastNormalization'[true]
%   and more defined in cnn_cifar.m

%setup;
% *********** Model Initialization *****************
% --------------------------------------------------
opts.datasetName = 'ILSVRC2012';
opts.datafn = @setup_imdb_imagenet;
[opts,varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data',opts.datasetName) ;
n=Ns;
%opts.imdb       = [];
opts.networkType = 'resnet' ;
opts.expDir     = fullfile('data','exp', ...
    sprintf('%s-%s-%d', opts.datasetName, opts.networkType , Ns)) ;

opts.batchNormalization = true ;
opts.nClasses = 1;
opts.batchSize = 64;
opts.numAugments = 1 ;
opts.numEpochs = 1;
opts.bn = true;
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'pixel'; % 'pixel' | 'image'
opts.gpus = []; 
opts.checkpointFn = [];
opts.border = [4 4 4 4];
[opts, varargin] = vl_argparse(opts, varargin) ;

%if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end
opts.numFetchThreads = 12 ;
%opts = vl_argparse(opts, varargin) ;
opts.expDir = fullfile('data','exp') ;
opts.gpus = [1];
opts.datasetName = ['pascalvoc2007_' cls];
opts.datafn = @setup_imdb_imagenet;


MTs = 'resnet';
n_exp = numel(Ns); 
if ischar(MTs), MTs = {MTs}; end; 
MTs = repmat(MTs, [1, n_exp]); 

expRoot = opts.expDir; 
opts.checkpointFn = @() plot_results(expRoot, opts.datasetName);

opts = vl_argparse(opts, varargin);
opts.expDir = fullfile(expRoot, ...
    sprintf('%s-%s-%d', opts.datasetName, MTs{1}, Ns)); 
if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
%                      Prepare model
% -------------------------------------------------------------------------

net = res_imagenet_init(n, 'nClasses', 1,...
                        'batchNormalization', opts.batchNormalization, ...
                        'networkType', opts.networkType, ...
                        'polyhedral', true, ...
                        'binary', true) ;
nettemp = load('/home/mlcv/Desktop/ResNet-Matconvnet/data/models/imagenet-resnet-101-dag.mat');
net2=dagnn.DagNN.loadobj(nettemp);

for id=1:size(net.params,2)-2
    if size(net.params(id).value(:)) == size(net2.params(id).value(:))
        net.params(id).value(:) = net2.params(id).value(:);
    else
        fprintf('skipping params %d\n', id)
    end
end

load(imdb_name);

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the image average (use either an image or a color)
%net.meta.normalization.averageImage = averageImage ;
net.meta.normalization.averageImage = rgbMean ;

% Set data augmentation statistics
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
clear v d ;

LearningRate = [0.0001*ones(1,10) 0.00001*ones(1,30)];
opts.train.weightDecay = 0.0005 ;
Numepochs = length(LearningRate);

for j=1:Numepochs,
     
    opts.train.learningRate = LearningRate(j);
  %  gpuDevice(1)
    
     if j<=5,
        % ----------------------------------------------
        %                  Learn
        % ----------------------------------------------
       
        trainfn = @cnn_train_dag_check;
        [net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
            'expDir', opts.expDir, ...
            net.meta.trainOpts, ...
            'gpus', opts.gpus, ...
            'batchSize',opts.batchSize,...
            'numEpochs',opts.numEpochs,...
            'val', find(imdb.images.set == 3), ...
            'derOutputs', {'loss', 1}, ...
            'checkpointFn', opts.checkpointFn, ...
            'learningRate', opts.train.learningRate, ...
            'weightDecay', opts.train.weightDecay) ;
       opts.numEpochs =  opts.numEpochs +1;
    else
        % ** Computing Polyhedral Cone Vertex ***
        net.mode = 'test' ;
        net.accumulateParamDers = 0;
        net.conserveMemory = 1;
        net.parameterServer = [];
        
        net.vars(end-6).precious = 1;
        net.move('gpu');
               
        load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
        optsTest.numThreads = 12;
        optsTest.imageSize = [224 224 3];
        optsTest.border = [32 32];
        optsTest.averageImage = rgbMean;
        optsTest.rgbVariance = [];
        optsTest.transformation = 'stretch';
        optsTest.numAugments = 1;
        
        pos_labels=find(imdb.images.label==1);
        test_it=ceil(length(pos_labels)/opts.batchSize);
        stest = 0;
        count = 0;
        batch_size = opts.batchSize;
        scores = [];
        for h=1:test_it,
            if h==test_it,
                batch_size = (length(pos_labels)-(test_it-1)*opts.batchSize);
            end
            clear imt
            for jj=1:batch_size,
                stest = stest + 1;
                imname = imdb.images.name{pos_labels(stest)};
                imfile = [imdb.imageDir '/' imname];
                imt{jj} = imfile;
            end
            imtemp = cnn_imagenet_get_batch(imt, optsTest, ...
                'prefetch', false, ...
                'transformation', 'none') ;
            count=count+batch_size;
            input = {'data', gpuArray(single(imtemp)), 'label', single(ones(1,batch_size))'};
            
            net.eval(input);
            output = net.vars(end-6).value;
            scorestemp = squeeze (output);
            scores = [scores scorestemp];
            
        end
        
        mu = mean(scores')';
        %%%
        net.mode = 'train';
        net.layers(377).block.updateMuCache = 0;
        net.layers(377).block.muCache = mu;
        gpuDevice(1)
        smodel = int2str(j-1);
        modelPath = fullfile(opts.expDir, ['net-epoch-' smodel '.mat']) ;
        nettemp = load(modelPath);
        net = dagnn.DagNN.loadobj(nettemp.net);
        
        trainfn = @cnn_train_dag_check;
        
        [net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
            'expDir', opts.expDir, ...
            net.meta.trainOpts, ...
            'gpus', opts.gpus, ...
            'batchSize',opts.batchSize,...
            'numEpochs',opts.numEpochs,...
            'val', find(imdb.images.set == 3), ...
            'derOutputs', {'loss', 1}, ...
            'checkpointFn', opts.checkpointFn, ...
            'learningRate', opts.train.learningRate, ...
            'weightDecay', opts.train.weightDecay) ;
        
        opts.numEpochs =  opts.numEpochs +1;
    end
        
    
end
    
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;
bopts.numAugments = opts.numAugments ; 

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case {'dagnn', 'resnet'}
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.label(batch) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'data', im, 'label', labels} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(temp, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;




    

