function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------
    
    cost = 0;
    costSq = 0;
    costSp = 0;
    costR = 0;
    grad = zeros(size(weightMatrix));

    l1 = weightMatrix * featureMatrix;
    l2 = l1-patches;
    l3 = l2.^2;
    
    f = sum(l3, 1);
    
    delta = l3;
    
    costSq = mean(f); % average sum-of-squares error term
    
    % costSp = lambda * sum(mean(sqrt(featureMatrix.^2 + epsilon)));
    
    costR = gamma * sum(sum(weightMatrix.^2));
    
    cost = costSq + costSp + costR;
    
    %gradient
    delta3 = 2*(delta);
    delta2 = eye(visibleSize)' * delta3;
    delta1 = weightMatrix' * delta2;
    
    
    grad1 = weightMatrix'*2*(weightMatrix * featureMatrix - patches);
    
    grad = delta1 + gamma * weightMatrix;
    
    grad = grad(:);
    
    

end