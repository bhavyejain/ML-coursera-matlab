function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

theta = zeros(size(X,2),1);

for i = 1:m
	X_t =  X(1:i, :);
	Y_t = y(1:i);
	theta = trainLinearReg(X_t,Y_t,lambda);
	error_train(i) = linearRegCostFunction(X_t,Y_t,theta,0);
	error_val(i) = linearRegCostFunction(Xval,yval,theta,0);
end

end