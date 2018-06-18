function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

pred = X*theta;
error = pred - y;
sqError = error.^2;

J = (1/(2*m))*sum(sqError) + (lambda/(2*m))*(sum(theta.^2) - (theta(1)^2));

s = sum(X.*error, 1);
s= s(:);
thetaReg = theta;
thetaReg(1) = 0;
grad = s/m + (lambda/m)*(thetaReg); 

end