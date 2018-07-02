function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X

[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

mu = (sum(X, 1)./m)';

sqDiff = (X - mu').^2;
sigma2 = (sum(sqDiff, 1)./m)';

end