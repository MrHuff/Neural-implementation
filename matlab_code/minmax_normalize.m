function [ Z ] = minmax_normalize(X)
%MINMAX_NORMAL Summary of this function goes here
%   Detailed explanation goes here

A = bsxfun(@minus, X,min(X));
Z = bsxfun(@rdivide, A, max(X)-min(X));



end

