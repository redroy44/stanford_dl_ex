%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
cost = params.lambda * sqrt((W*x).^2.0+ params.epsilon);
cost = sum(cost(:));
delta = W'*W*x - x;
cost = cost + 0.5*trace(delta'*delta) ./ params.m;

Wgrad  = W * (2*delta) * x' + 2 * (W*x)*delta';
Wgrad = 0.5 * Wgrad ./params.m + params.lambda * ((W*x).^2.0+ params.epsilon).^(-0.5).*(W*x) * x';

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);