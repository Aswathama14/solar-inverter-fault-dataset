%% load_dataset_matlab.m
% =========================================================================
% Grid-Tied Solar Inverter Multi-Class Fault Detection Dataset
% 5-Class: Normal, SLG(A-G), SLG(B-G), SLG(C-G), Three-Phase Short
%
% Paper: Patel et al. (2026), IEEE GreenTech Conference
% DOI: 10.1109/GreenTech68285.2026.11471570
% =========================================================================

%% Load raw CSV
data = readtable('data/inverter_fault_dataset.csv');
fprintf('Loaded %d rows\n', height(data));

%% Extract features and labels
X = table2array(data(:, 2:12));   % 11 features: Va,Vb,Vc,Ia,Ib,Ic,Id,Iq,P,Q,THD
y = data.label;                    % Fault labels (0-4)

%% Display class distribution
class_names = {'Normal', 'SLG(A-G)', 'SLG(B-G)', 'SLG(C-G)', '3-Phase Short'};
fprintf('\nClass Distribution:\n');
for c = 0:4
    fprintf('  Label %d (%s): %d samples\n', c, class_names{c+1}, sum(y == c));
end

%% Z-score normalization (compute stats on training split only)
train_ratio = 0.70;
n_train = round(size(X, 1) * train_ratio);

mu = mean(X(1:n_train, :));
sigma = std(X(1:n_train, :));
X_norm = (X - mu) ./ sigma;

%% Sliding window segmentation
W = 100;   % Window size: 100 samples = 10 ms at 10 kHz
S = 20;    % Step size: 20 samples = 2 ms

n_win = floor((size(X_norm, 1) - W) / S);
X_seq = zeros(n_win, W, 11);
y_seq = zeros(n_win, 1);

for i = 1:n_win
    idx = (i-1)*S + 1;
    X_seq(i, :, :) = X_norm(idx:idx+W-1, :);
    % Majority vote for window label
    window_labels = y(idx:idx+W-1);
    [~, y_seq(i)] = max(histcounts(window_labels, -0.5:1:4.5));
    y_seq(i) = y_seq(i) - 1;  % Convert to 0-indexed
end

fprintf('\nCreated %d sliding windows\n', n_win);
fprintf('X_seq shape: [%d, %d, %d]\n', size(X_seq));

%% Train/test split
n_train_win = round(n_win * train_ratio);
X_train = X_seq(1:n_train_win, :, :);
y_train = y_seq(1:n_train_win);
X_test = X_seq(n_train_win+1:end, :, :);
y_test = y_seq(n_train_win+1:end);

fprintf('Train: %d windows, Test: %d windows\n', size(X_train, 1), size(X_test, 1));

%% If you use this dataset, please cite:
% D. P. Patel, I. P. Pathak, D. Roach, and N. Yilmazer,
% "Robust Deep Learning Models for Fault Detection in Grid-Tied Solar
% Inverters: A Comparative Study of LSTM, Bi-LSTM, and CNN-LSTM
% Architectures," in Proc. IEEE Green Tech. Conf., Boulder, CO, USA, 2026.
% DOI: 10.1109/GreenTech68285.2026.11471570
