function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

for c = 1:num_labels

	y_t = (y == c);   % Create a one-vs-all labelled matrix, with logical 1 for class c and 0 for the rest
	
	initial_theta = zeros(n+1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 100);
	
	[theta, cost] = fmincg (@(t)(lrCostFunction(t, X, y_t, lambda)), initial_theta, options);  % Using fmincg because of large number of parameters																							    
	
	theta = theta(:);     % returns a column vector
	all_theta(c,:) = theta;

end

end