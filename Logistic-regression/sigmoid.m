function g = sigmoid(z)
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

exponent = exp(-z) + 1;
g = 1./exponent;

end
