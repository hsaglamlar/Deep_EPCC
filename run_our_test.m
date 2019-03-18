

net.vars(end-3).precious = 1;
net.move('gpu') ;
input = {'data', gpuArray(single(rand(224,224,3,32))), 'label', round(rand(1,32)*9)+1};
net.eval(input);
output = net.vars(end-3).value;