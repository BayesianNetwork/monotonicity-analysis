%% This program is used to implement PPLS and PPCA.
%% Sometimes the reconstruction results of PPLS are unsatisfactory, so it should be repeated several times.

%% Program initialization
clear all
close all
clc

%% Load Data
load('Data.mat')
X = DataX;
Y = DataY;

%% Probabilistic PLS Model
%% If the results are unsatisfactory, please try again.

model = ppls_rand(X, Y, 2, 100, 1e-6);  % Random initialization
% model = ppls_plsinit(X, Y, 2, 100, 1e-6);  % Initialization using PLS

Y_hat = (X - model.Xmean) * model.B + model.Ymean;
rmse = sqrt(mean((Y_hat(:, 1) - Y(:, 1)).^2))

figure
plot(Y(:, 1), 'b')
hold on
plot(Y_hat(:, 1), 'r')
xlabel('Y1', 'FontSize', 50);
ylabel('Value', 'FontSize', 50);

%% Probabilistic PCA Model
[coeff, score, pcvar, mu] = ppca(Y, 2);
Y_rec = score * coeff' + repmat(mu, size(X,1), 1);
rmse = sqrt(mean((Y_rec(:, 1) - Y(:, 1)).^2))

figure
plot(Y(:, 1), 'b')
hold on
plot(Y_rec(:, 1), 'r')
xlabel('Y1', 'FontSize', 50);
ylabel('Value', 'FontSize', 50);
