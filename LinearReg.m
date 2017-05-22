%==================================================================
%Linear Regression
%       This sample code show how we can predict housing price
%       based on the square foot of the house
%
% X = Vector of square foot of houses
% y = Vector of the prices of the house corresponding to vector X
% m = Sample size 
% n = Number of features
%
%==================================================================
% Initialization
%==================================================================
clear ; close all; 
iterations = 4000;             % number of Gradient Descent iterations
alpha = 0.01;                 % alpha setting

testsize1 = 3000;
testbr1 = 3;
testsize2 = 4500;
testbr2 = 5;

data = load('ex1data2.txt');
data = sortrows(data);
X = data(:,1)./1000;    % load sizes and scale down to one thousandth
y = data(:,3)./1000;    % load prices and scale down to one thousandth
m = length(y);          % sample size

%==================================================================
% Initialize theta (fitting) and add column of 1's to data set
%==================================================================
X = [ones(m, 1), X];      % Add intercept term to x = X(0)
theta = zeros(2, 1);      % initialize fitting parameters

%==================================================================
% Sort and print sample data set
%==================================================================

%fprintf(' x = %.0f %0f %0f, y = %.0f \n', [sortrows(X) y]');
%******************************************************************
%==================================================================
% ********* FIRST TEST USING JUST HOUSING SIZE **********
%==================================================================
%******************************************************************

%==================================================================
% Plot data Size vs. Prices
%==================================================================
fprintf('Plotting Data using Housing Sizes ...\n')
figure;                           % open a new figure window
plot(X(:,2), y, 'rx', 'MarkerSize', 10); % Plot the size and price
xlabel('Square Foot of Houses');  % Set the x axis label
ylabel('Price of Houses');        % Set the y axis label

%==================================================================
% Test Cost Function using initial theta of zeros
%==================================================================
%fprintf('\nTesting the cost function ...\n')
%J = computeCost(X, y, theta);   % compute and display initial cost
%fprintf('With theta = [0 ; 0; 0]\nCost computed = %f\n', J);

%==================================================================
% Run Gradient Descent to determine optimal theta
%==================================================================
fprintf('\nRunning Gradient Descent for Univariate Housing Sizes ...\n')
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent with Univariate Housing Sizes:\n');
fprintf('%f\n', theta);

%==================================================================
% Plot linear fit line 
%==================================================================
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression');
hold off % don't overlay any more plots on this figure

%==================================================================
% Plot Cost Function
%==================================================================
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%==================================================================
% Predict housing prices for (3000sf 3 br) and (4000sf 4bd)
%==================================================================
predict1 = [1, testsize1/1000]*theta;
fprintf('For housing size = %f using Univariate, we predict a price of %f\n',...
    testsize1, predict1*1000);
predict2 = [1,testsize2/1000] * theta;
fprintf('For housing size = %f using Univariate, we predict a price of %f\n',...
    testsize2, predict2*1000);
%******************************************************************
%==================================================================
% ********* SECOND TEST USING HOUSING SIZE AND BEDROOMS  **********
%==================================================================
%******************************************************************
X2 = [X,data(:,2)];      % load bedrooms
theta2 = zeros(3, 1);     % initialize fitting parameters

%==================================================================
% Plot data Size vs. Housing size * Bedrooms
%==================================================================
fprintf('Plotting Data using Size*BR ...\n')
figure;                           % open a new figure window
plot(X2(:,2).*X2(:,3), y, 'rx', 'MarkerSize', 10); % Plot the size and price
xlabel('Square Foot X BR of Houses');  % Set the x axis label
ylabel('Price of Houses');        % Set the y axis label

%==================================================================
% Run Gradient Descent to determine optimal theta
%==================================================================
fprintf('\nRunning Gradient Descent using Multivariate (Size and Bedrooms) ...\n')
[theta2, J_history2] = gradientDescent(X2, y, theta2, alpha, iterations);
fprintf('Theta found by gradient descent with Multivariate:\n');
fprintf('%f\n', theta2);

%==================================================================
% Plot Cost Function
%==================================================================
% Plot the convergence graph
figure;
plot(1:numel(J_history2), J_history2, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%==================================================================
% Predict housing prices for (3000sf 3 br) and (4000sf 4bd)
%==================================================================
predict1 = [1, testsize1/1000, testbr1]*theta2;
fprintf('For housing size = %f and %f br using Multivariate, we predict a price of %f\n',...
    testsize1, testbr1, predict1*1000);
predict2 = [1,testsize2/1000, testbr2] * theta2;
fprintf('For housing size = %f and %f br using Multivariate, we predict a price of %f\n',...
    testsize2, testbr2, predict2*1000);
    
%******************************************************************
%==================================================================
% ********* THIRD TEST USING FEATURE SCALING WITH NORMALIZATION ***
%==================================================================
%******************************************************************
% Load Data
data = load('ex1data2.txt');
X3 = data(:, 1:2);
y = data(:, 3);
theta3 = zeros(3, 1);

%==================================================================
% Scale features and set them to zero mean
%==================================================================
% Mormalizing Features
[X3 mu sigma] = featureNormalize(X3);
% Add intercept term to X
X3 = [ones(m, 1) X3];

%==================================================================
% Run Gradient Descent to determine optimal theta
%==================================================================
fprintf('\nRunning Gradient Descent using Normalization (Size and Bedrooms) ...\n')
[theta3, J_history3] = gradientDescent(X3, y, theta3, alpha, iterations);
fprintf('Theta found by gradient descent with Normalization:\n');
fprintf('%f\n', theta3);

%==================================================================
% Plot Cost Function
%==================================================================
% Plot the convergence graph
figure;
plot(1:numel(J_history3), J_history3, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%==================================================================
% Predict housing prices for (3000sf 3 br) and (4000sf 4bd)
%==================================================================
element = [1,(testsize1 - mu(1))/sigma(1),(testbr1 - mu(2))/sigma(2)];
price = element * theta3;
fprintf('For housing size = %f and %f br using Normalization, we predict a price of %f\n',...
    testsize1, testbr1, price);
element = [1,(testsize2 - mu(1))/sigma(1),(testbr2 - mu(2))/sigma(2)];
price = element * theta3;
fprintf('For housing size = %f and %f br using Normalization, we predict a price of %f\n',...
    testsize2, testbr2, price);

%******************************************************************
%==================================================================
% ********* FOURTH TEST USING NORMAL EQUATIONS ***
%==================================================================
%******************************************************************

fprintf('Solving with normal equations...\n');

% Load Data
data = csvread('ex1data2.txt');
X4 = data(:, 1:2);
y4 = data(:, 3);

% Add intercept term to X
X4 = [ones(m, 1) X4];

% Calculate the parameters from the normal equation
theta4 = normalEqn(X4, y4);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta4);
fprintf('\n');


%==================================================================
% Predict housing prices for (3000sf 3 br) and (4000sf 4bd)
%==================================================================
price = 0; % You should change this
element = [1,testsize1,testbr1];
price = element * theta4;
fprintf(['Predicted price of a %f sq-ft, %f br house ' ...
         '(using normal equations):\n $%f\n'], testsize1, testbr1, price);
element = [1,testsize2,testbr2];
price = element * theta4;
fprintf(['Predicted price of a %f sq-ft, %f br house ' ...
         '(using normal equations):\n $%f\n'], testsize2, testbr2, price);
