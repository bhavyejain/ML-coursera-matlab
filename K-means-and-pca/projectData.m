function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors

Z = zeros(size(X, 1), K);

Ureduce = U(:, 1:K);
Z = X * Ureduce;

end
