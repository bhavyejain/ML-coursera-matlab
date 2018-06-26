function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
[m n] = size(X);

centroids = zeros(K, n);

%% iterative implementation
% for i = 1:K
%     centroids(i,:) = mean(X(idx==i,:));
% end

%% vectorized implementation

k = 1:K;                   % a vector from 1 to K (the number of clusters): [1 2 3 .... K]

logic2D = (idx == k);      % a m*K logic 1 matrix. In each row (training example i), 
                           % logic 1 in the column corresponding to the cluster number in idx.
                           % eg: [ 0 0 1 ; 0 1 0 ; 0 0 1 ; 1 0 0 ; ......] 

logic3D = permute(logic2D, [3 2 1]);     % convert logic2D to a (1 x K x m) 3D matrix, making
										 % the matrix lie in the XZ plane

X_3D = permute(X, [2 3 1]);				 % convert X to a (2 x 1 x m) 3D matrix, making the 
										 % matrix lie in the YZ plane, so that each example
										 % is a column vector, and examples are placed in 
										 % layers one behind another.

X_sorted = bsxfun(@times, X_3D, logic3D);   % create a (2 x K x m) matrix. This has each example
											% of X_3D placed into one of the K columns,
											% corresponding to cluster they belong to, rest of  
											% the elements in the layer being zeros. There are
											% m layers of (2 x K) matrices.

centroids = (sum(X_sorted, 3)./sum(logic3D, 3))';   % sum the elements along the Z axis, to get
													% a (2*K) matrix, with each column being the
													% sum of the examples belonging to that cluster.
													% Sum the logic3D matrix in a similar manner to 
													% get each column as the number of examples
													% belonging to the corresponding cluster.
													% Element wise division to compute the mean. 

end

