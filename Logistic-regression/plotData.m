function plotData(X, y)

%PLOTDATA Plots the data points X and y into a new figure 

% Create New Figure
figure; hold on;

pos = find(y==1);    %indices of positive examples
neg = find(y==0);    %indices of negative examples

plot(X(pos, 1), X(pos, 2), 'r+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'bo', 'LineWidth', 2, 'MarkerSize', 7);

hold off;

end