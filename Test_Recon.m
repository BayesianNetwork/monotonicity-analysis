%% This program is used to evaluate the reconstruction performance of the PI-PQM algorithm on simulated data.

%% Program initialization
clear all
close all
clc

%% Load the data and set the model parameters.
load('Data.mat')
Num_Ret = 2;
[U_Coeff, V_Coeff, Y_Sigma, Y_Tao, X_Sigma, X_Tao, Y_Error, X_Error, S_Mu, S_Sig] = Pi_QPR(DataY, DataX, Num_Ret);

%% Validate the monotonicity of the reconstruction results.
Y_Mu = mean(DataY);
M_Matrix = diag(std(DataY));
YRec = (M_Matrix * U_Coeff * S_Mu)' + Y_Mu;
figure
plot(DataY(:, 1), 'g')
xlabel('Y1', 'FontSize', 50);
ylabel('Value', 'FontSize', 50);
figure
plot(YRec(:, 1), 'r')
xlabel('Y1_Rec', 'FontSize', 50);
ylabel('Value', 'FontSize', 50);

%% Evaluate the reconstruction performance of the output data.
x1 = min(DataY(:, 1)) : 0.01 : max(DataY(:, 1));
x2 = min(DataY(:, 2)) : 0.01 : max(DataY(:, 2));
[X1, X2] = meshgrid(x1, x2);
X = [X1(:) X2(:)];

YMu = mean(YRec);
YCov = cov(YRec);

YData = mvnpdf(X, YMu, YCov);
YData = reshape(YData, length(x2), length(x1));

figure
plot(DataY(:, 1), DataY(:, 2), 'g^', 'MarkerSize', 2, 'MarkerFaceColor','r');
hold on

contour(x1, x2, YData, 3, '--r', 'LineWidth', 5);
xlabel('Y1', 'FontSize', 50);
ylabel('Y2', 'FontSize', 50);

%% Validate the range of the results corresponding to the input variables
X_Mu = mean(DataX);                                                        
N_Matrix = diag(std(DataX));                                               
XRec = (N_Matrix * V_Coeff * S_Mu)' + X_Mu;
figure
scatter(XRec(:, 1), XRec(:, 2), 'r')
hold on
scatter(XRec(:, 2), XRec(:, 3), 'g')
hold on
scatter(XRec(:, 1), XRec(:, 3), 'k')

%% Evaluate the reconstruction performance of the input data.
x1 = min(DataX(:, 1)) : 0.02 : max(DataX(:, 1));
x2 = min(DataX(:, 2)) : 0.02 : max(DataX(:, 2));
x3 = min(DataX(:, 3)) : 0.02 : max(DataX(:, 3));

% X1 and X2
[X1, X2] = meshgrid(x1, x2);
X = [X1(:) X2(:)];
XMu = mean(XRec(:, [1, 2]));
XCov = cov(XRec(:, [1, 2]));
XData = mvnpdf(X, XMu, XCov);
XData = reshape(XData, length(x2), length(x1));
figure
plot(DataX(:, 1), DataX(:, 2), 'g^', 'MarkerSize', 2, 'MarkerFaceColor','r');
hold on
contour(x1, x2, XData, 3, '--r', 'LineWidth', 5);
xlabel('X1', 'FontSize', 50);
ylabel('X2', 'FontSize', 50);

% X1 and X3
[X1, X3] = meshgrid(x1, x3);
X = [X1(:) X3(:)];
XMu = mean(XRec(:, [1, 3]));
XCov = cov(XRec(:, [1, 3]));
XData = mvnpdf(X, XMu, XCov);
XData = reshape(XData, length(x3), length(x1));
figure
plot(DataX(:, 1), DataX(:, 3), 'g^', 'MarkerSize', 2, 'MarkerFaceColor','r');
hold on
contour(x1, x3, XData, 3, '--r', 'LineWidth', 5);
xlabel('X1', 'FontSize', 50);
ylabel('X3', 'FontSize', 50);

% X2 and X3
[X2, X3] = meshgrid(x2, x3);
X = [X2(:) X3(:)];
XMu = mean(XRec(:, [2, 3]));
XCov = cov(XRec(:, [2, 3]));
XData = mvnpdf(X, XMu, XCov);
XData = reshape(XData, length(x3), length(x2));
figure
plot(DataX(:, 2), DataX(:, 3), 'g^', 'MarkerSize', 2, 'MarkerFaceColor','r');
hold on
contour(x2, x3, XData, 3, '--r', 'LineWidth', 5);
xlabel('X2', 'FontSize', 50);
ylabel('X3', 'FontSize', 50);

