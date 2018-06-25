function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
[m n] = size(X);

centroids = zeros(K, n);

for i = 1:K
    centroids(i,:) = mean(X(idx==i,:));
end

end

