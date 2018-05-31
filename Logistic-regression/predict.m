function p = predict(theta, X)

%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 using learned logistic 
%   regression parameters theta

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

prob = sigmoid(X*theta);
p(find(prob >= 0.5)) = 1;
p(find(prob < 0.5)) = 0;

end