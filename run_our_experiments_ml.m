% run_our_experiments([101], 'gpus', [1]);
function run_our_experiments_ml(Ns, varargin)
% Usage example: run_experiments([18 34 50 101 152], 'gpus', [1 2 3 4]); 
% On you own dataset: run_experiments([18 34 50 101],'datasetName',...
% 'reflectance', 'datafn', @setup_imdb_reflectance, 'gpus', [1 2]);
% Options: 
%   'expDir'['exp'], 'bn'[true], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   'meanType'['image'], 'whitenData'[true], 'contrastNormalization'[true]
%   and more defined in cnn_cifar.m

% setup;
opts.expDir = fullfile('data','exp') ;
opts.bn = true;
opts.meanType = 'image';
opts.whitenData = true;
opts.contrastNormalization = true; 
opts.border = [4 4 4 4];
opts.gpus = [1];
opts.datasetName = 'coco';
opts.datafn = @setup_imdb_imagenet;
opts.nClasses = 80;
opts.batchSize = 64;

opts = vl_argparse(opts, varargin); 

MTs = 'resnet';
n_exp = numel(Ns); 
if ischar(MTs), MTs = {MTs}; end; 
MTs = repmat(MTs, [1, n_exp]); 


expRoot = opts.expDir; 
opts.checkpointFn = @() plot_results(expRoot, opts.datasetName);

for i=1:n_exp, 
  opts.expDir = fullfile(expRoot, ...
    sprintf('%s-%s-%d', opts.datasetName, MTs{i}, Ns(i))); 
  [net,info] = res_our_coco_ml(Ns(i), opts); 
end
