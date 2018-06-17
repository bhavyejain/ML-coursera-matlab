function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% ========= forward propagataion ===========
a_2 = sigmoid(Theta1 * [ones(m, 1) X]');    % Activation of Layer 2
a_3 = sigmoid(Theta2 * [ones(1, size(a_2, 2)); a_2]);   % Activation of output layer (Layer 3)

%% calculate cost
s = 0;
y_t = zeros(num_labels,1);
for i = 1:m
	h = a_3(:, i);   % get the hypothesis vector 
    y_t(y(i)) = 1;   % generate the label vector

	for k = 1:num_labels
		s = s + ( (y_t(k)*log(h(k))) + ((1 - y_t(k))*log(1 - h(k))) );
	end;

	y_t(y(i)) = 0;   % reset vector y_t;
end;

theta1Sq = Theta1(:, (2 : size(Theta1, 2))).^2;
theta2Sq = Theta2(:, (2 : size(Theta2, 2))).^2;

reg = sum(sum(theta1Sq)) + sum(sum(theta2Sq));   % regularization term

J = (-1/m)*s + (lambda/(2*m))*reg;   % cost function

%% =========== back propagataion ==============

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t = 1:m
	a1 = [1 X(t , :)]';
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	y_t(y(t)) = 1;  % generate label vector
    
    % calculate errors
    del3 = a3 - y_t;
    del2 = (Theta2' * del3).*(a2.*(1 - a2));
    del2 = del2(2:end);  % removing error of bias term 

    delta_2 = delta_2 + (del3 * a2');  % accumulate gradient 2
    delta_1 = delta_1 + (del2 * a1');  % accumulate gradient 1

    y_t(y(t)) = 0;
end;

% calculate regularization term for Theta1_grad
reg1 = [zeros(size(Theta1,1), 1) (lambda/m) * Theta1(:, (2:end))];   

% calculate regularization term for Theta2_grad
reg2 = [zeros(size(Theta2,1), 1) (lambda/m) * Theta2(:, (2:end))];

% calculate regularized partial derivatives
Theta1_grad = (1/m)*delta_1 + reg1;
Theta2_grad = (1/m)*delta_2 + reg2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end