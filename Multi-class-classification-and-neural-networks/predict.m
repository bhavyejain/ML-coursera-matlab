function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a_2 = sigmoid(Theta1 * [ones(m, 1) X]');    % Activation of Layer 2

a_3 = sigmoid(Theta2 * [ones(1, size(a_2, 2)); a_2]);   % Activation of output layer (Layer 3)

[max_val, lab_t] = max(a_3, [], 1);  % get column wise max and corresponding row number (here row number equals the label)
p = lab_t(:);

end