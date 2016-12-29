function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

[numFeatures, numExamples] = size(x);

%%% YOUR CODE HERE %%%
% mean removal - not needed - patches are individually normalized
%avg = mean(x, 1);
%x = x - repmat(avg, numFeatures, 1);

% sigma
sigma = x * x' / numExamples;

% svd
[U,S,V] = svd(sigma);

% xRot
xRot = U' * x;

Z = U * diag(1./sqrt(diag(S) + epsilon)) * xRot;









