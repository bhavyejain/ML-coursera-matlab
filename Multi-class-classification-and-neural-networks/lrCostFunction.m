function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

pred = sigmoid(X*theta);

J = (1/m)*sum((-y.*log(pred)) - ((1-y).*log(1-pred))) + (lambda/(2*m))*(sum(theta.^2)-(theta(1)^2));

error = pred - y;
reg = (lambda/m)*theta;						% Calculate the regularization term
reg(1) = 0;   								% We dont want to regularize theta(1) 
grad = ((1/m)*sum((X.*error),1))' + reg;
grad = grad(:);

end
