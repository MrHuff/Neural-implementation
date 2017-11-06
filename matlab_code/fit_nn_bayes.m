function [ metric ] = fit_nn_bayes(neural_net_para,X,Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net = initialize_nn(neural_net_para.list,... 
                    neural_net_para.layer_func,...
                    neural_net_para.output_func,...
                    neural_net_para.learning_rate);
                
[net,tr] = trainbr(net,X,Y);

metric = tr.best_perf;


end

