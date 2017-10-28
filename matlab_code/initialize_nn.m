function [net] = initialize_nn(list, layer_func,output_func,learning_rate)
%INITIALIZE_NN Summary of this function goes here
%   Detailed explanation goes here



net = patternnet(list);

for i=1:(net.numLayers-1)
  net.layers{i}.transferFcn = layer_func;
end

%net.inputs{1}.size = input_size;

net.layers{end}.transferFcn = output_func;
net.divideFcn = 'divideind';
net.trainParam.lr = learning_rate;

%view(net)



end

