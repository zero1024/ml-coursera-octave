function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


g = X * theta;
predictions = sigmoid(g);
j1 = - y .* log(predictions);
j2 = (1 - y) .* log(1 - predictions);
penalty = sum(theta(2:size(theta,1)) .^ 2) * (lambda/(2 * m));
J = (sum(j1 - j2)/m) + penalty;

gradPenalty = [0 ; theta(2:size(theta,1))] .* (lambda/m);
grad = ((X' * (predictions - y)) ./ m) + gradPenalty;



% =============================================================

end
