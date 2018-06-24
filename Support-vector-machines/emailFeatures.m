function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices 

n = 1899;

x = zeros(n, 1);

for i = 1:size(word_indices)
	x(word_indices(i)) = 1;
end;

end
