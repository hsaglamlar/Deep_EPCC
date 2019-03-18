method = 'polyhedral_bc';

if strcmp(method, 'polyhedral')
    run_our_experiments([101], 'gpus', [1]);
elseif strcmp(method, 'softmax')
    run_our_experiments_softmax([101], 'gpus', [1]);
elseif strcmp(method, 'polyhedral_bc')
    run_our_experiments_bc([101], 'gpus', [1]);
elseif strcmp(method, 'polyhedral_ml')
    run_our_experiments_ml([101], 'gpus', [1]);
end
