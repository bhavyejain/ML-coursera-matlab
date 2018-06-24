function [C, sigma] = dataset3Params(X, y, Xval, yval)
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of
%   the optimal C and sigma based on a cross-validation set.

C = 0.3;
sigma = 0.1;

minError = 1000000;

c = 0.01;
sig = 0.01;

while c <= 30,
	while sig <= 30,

		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));      % train the model 
		predictions = svmPredict(model, Xval);        % compute predictions 
		error = mean(double(predictions ~= yval));    % compute prediction error

		if(error <= minError),
			C = c;
			sigma = sig;
			minError = error;
		end;

		sig = sig * 3;
		if(sig == 0.09)
			sig = 0.1;
	    end;
		if (sig == 0.9)
			sig = 1;
		end;

	end;

	sig = 0.01;

	c = c * 3;
	if(c <= 0.1 && c >= 0.08)
		c = 0.1;
	end;
	if (c <= 1 && c >= 0.8)
		c = 1;
	end;

end;

end