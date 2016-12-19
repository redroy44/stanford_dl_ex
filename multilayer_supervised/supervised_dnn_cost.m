function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
netSize = numel(ei.layer_sizes) + 1;
hAct = cell(netSize, 1);
deltaStack = cell(netSize, 1);
gradStack = cell(numHidden+1, 1);
m = size(data, 2);
%% forward prop
%%% YOUR CODE HERE %%%
hAct{1}.a = data;
hAct{1}.z = data;

for i = 1:netSize-1
  hAct{i+1}.z = stack{i}.W * hAct{i}.a + repmat(stack{i}.b, 1, m);
  hAct{i+1}.a = sigmoid(hAct{i+1}.z);
end

% softmax
h = exp(hAct{end}.z);
h_sum = sum(h, 1);
h_sum = repmat(h_sum, ei.output_dim, 1);
h = h ./ h_sum;

%% return here if only predictions desired.
if po
  pred_prob = h;
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y = labels;
c = 1:ei.output_dim;
I = bsxfun(@eq, y, c)';
ceCost = -sum(sum(I.*log(h))); % + reg term

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
deltaStack{end} = -(I - h);

for i = netSize-1:-1:1
  deltaStack{i} = stack{i}.W'*deltaStack{i+1} .* grad_sigmoid(hAct{i}.z);
end

for i = 1:numHidden+1
  gradStack{i}.W = deltaStack{i+1} * hAct{i}.a' + ei.lambda.*stack{i}.W;
  gradStack{i}.b = sum(deltaStack{i+1}, 2);
end


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
pCost = 0;
for i = 1:netSize-1
  pCost = pCost + ei.lambda/2*sum(sum(stack{i}.W.^2));
end



%% reshape gradients into vector
[grad] = stack2params(gradStack);
pred_prob = 0;
cost = ceCost + pCost;

end



