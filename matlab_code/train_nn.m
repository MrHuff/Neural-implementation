input_range = 'A1:ET10'; %Needs to be adapted
output_range = 'A12:ET21'; %Needs to be adapted

%Read data

train_first = {};
output = {};
nfolds=10;
for i = 1:1:nfolds;
    train_data= xlsread('data.xlsx',i,input_range); %Rename data set data
    output_data = xlsread('data.xlsx',i,output_range);
    train_first{end+1} = train_data;
    output{end+1} = output_data;
end

features = cell2mat(train_first);
target = cell2mat(output);
block_size = length(target)/nfolds;

%hyperparamter set:
hyper.neuron_size = [10:25];

hyper.layers = [5:10];
hyper.learning_rate = [0.01,0.03,0.05];
hyper.transfer = {'tansig'};




%%
%initialize neural net


total_range = 1:length(features);
X = tonndata(features);
Y = tonndata(target);
%X_test = tonndata(test_col{i});
precisions = [];
nr_iterations = 25;
confusion_mats = {};
nets = {};
for j = 1:1:nr_iterations
    tmp = datasample(hyper.layers,1);
    tmp_n = datasample(hyper.neuron_size,1);
    tmp_h = datasample(hyper.learning_rate,1);
    tmp_trans = datasample(hyper.transfer,1);
    layers = tmp(1);
    neuron_size = tmp_n(1);
    learning_rate = tmp_h(1);
    transfer = tmp_trans{1};
    net = initialize_nn(ones(1,layers)*neuron_size,transfer,'softmax',learning_rate);
    for i=1:nfolds; % first fold as test-set
        test_index = ((i-1)*block_size+1):i*block_size;
        v_index = datasample( [setdiff([1:nfolds],i)],1);
        validation_index = ((v_index-1)*block_size+1):v_index*block_size;
        train_index = setdiff(total_range, [test_index,validation_index]);
        net.divideParam.trainInd = train_index;
        net.divideParam.valInd = validation_index;
        net.divideParam.testInd = test_index;
        net = train(net, X,Y);
        %outputs = net(X_test);
        %figure, plotconfusion(test_output_col{i},outputs);
    end
    predicitions = net(features);
    [precision,mats] = confusion(target,predicitions);
    confusion_mats{end+1} = mats;
    precisions(end+1) = precision;
    nets{end+1} = net;
    
end


[best,index_best] = min(precisions);
best_net = nets{index_best};
confusion_best = confusion_mats{index_best};










%%
%making a single value prediction
    X_test = tonndata([293;
182;
218;
179;
131;
284;
224;
243;
143;
360
]);
   % outputs = net(X_test);
    
%Old code 
%{
%training_col = {};
%output_col = {};
%test_col = {};
%test_output_col = {};
%validation_col_features = {};
%validation_col_output = {};
%validation_indexes = [];
for i=1:1:nfolds; % first fold as test-set
    copy_train = train_first;
    copy_output = output;
    test_col{i} = train_first{i};
    test_output_col{i} = output{i};
    copy_train{i}=[];
    copy_output{i}=[];
    %validation_index = datasample( [setdiff([1:10],i)],1);
    %validation_col_features{end+1} = copy_train{validation_index}; 
    %validation_col_output{end+1} = copy_output{validation_index}; 
    %copy_train{validation_index} = [];
    %copy_output{validation_index}=[];
    %validation_indexes(end+1) = validation_index;
    training_col{i} = cell2mat(copy_train);
    output_col{i} = cell2mat(copy_output);
    
end
%}



