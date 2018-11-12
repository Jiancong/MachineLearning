function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples 97
J_history = zeros(num_iters, 1);

%disp("theta before size:"), disp(size(theta)) # 2x1

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta1 = theta(1,:) # 1x1
    theta2 = theta(2,:) # 1x1
    
    # X*theta 97x1, y:97x1, X(:,1):97x1
    theta1 = theta1 - 1/m * alpha * sum((X*theta - y).*X(:, 1)) %1x1
    theta2 = theta2 - 1/m * alpha * sum((X*theta - y).*X(:, 2)) %1x1

    theta = [theta1; theta2]
    #disp('theta size:'), disp(size(theta))

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
