function [ metric ] = fit_nn_bayes(neural_net_para,X,Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

architecture = neural_net_para.node_size*ones(1, neural_net_para.layer_size);

net = initialize_nn(architecture,... 
                    char(neural_net_para.activation),...
                    'softmax',...
                    neural_net_para.lr);
                
[net,tr] = train(net,X,Y);



metric = tr.best_perf;


end

