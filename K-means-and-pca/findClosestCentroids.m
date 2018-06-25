function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i = 1 : size(idx),
	x = X(i, :);
	norm = sum(((centroids - x).^2), 2);
	[m, j] = min(norm);
	idx(i) = j;
end

end