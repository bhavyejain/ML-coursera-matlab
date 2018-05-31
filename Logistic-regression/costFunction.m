function [J, grad] = costFunction(theta, X, y)

%COSTFUNCTION Computes cost and gradient for logistic regression

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

pred = sigmoid(X*theta);

J = (1/m)*sum((-y.*log(pred)) - ((1-y).*log(1-pred)));

error = pred - y;
grad = ((1/m)*sum((X.*error),1))';

end