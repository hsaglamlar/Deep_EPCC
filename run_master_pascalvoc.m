clear all;
close all; clc;
VOCinit;
method = 'polyhedral_bc';
for i=1:20,
    cls = VOCopts.classes{i};
    imdb_name = ['/home/mlcv/yedekdisk/pami_experiments/VOCdevkit/pascalvoc_imdb_' cls '.mat'];
    run_our_experiments_pascalvoc([101], imdb_name, cls, 'gpus', [1]);
end

