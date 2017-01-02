%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear all;
addpath(genpath('../common/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
params.m=10000; % num patches
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 50; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity

DEBUG = false;
if DEBUG
params.m=1000; % num patches
params.numFeatures = 10; % number of filter banks to learn
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
end
% Load MNIST data set
data = loadMNISTImages('../common/train-images-idx3-ubyte');

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
% Step 2) Apply ZCA
[patches, U, S] = zca2(patches);
% Step 3) Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;
options.Display = 'iter';
options.DerivativeCheck = 'off'; 
options.outputFcn = @showBases;

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);


if DEBUG
[cost, grad] = softICACost(randTheta, x, params);

numgrad = computeNumericalGradient( @(theta) softICACost(theta, x, params), randTheta );
% Uncomment the blow line to display the numerical and analytic gradients side by side
disp([numgrad grad]); 
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf('Gradient difference: %g\n', diff);
return
end

% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); % Use x or xw

% display result
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');
