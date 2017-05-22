function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));


num_features = size(X,2);

for x=1:num_features,
  mu(x) = mean(X(:,x));
  sigma(x) = std(X(:,x));
  X_norm(:,x) = (X_norm(:,x)-mu(x))/sigma(x);
end;

end
