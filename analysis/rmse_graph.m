% rmse_graph.m
% Pools RMSE results across all stored datasets and plots RMSE vs pose distance
% for all local methods and FFT lookup
clear; clc;

% Load Results
resultsFile    = 'C:\Users\wilso\RifeResearch\benchmark_results1.mat';
fftResultsFile = 'C:\Users\wilso\RifeResearch\fft_results.mat';
load(resultsFile, 'allResults');
load(fftResultsFile, 'fftResults');

% Pool Local Methods
localMethods = {'Splat','NN','Linear','BilinearSplat'};
nLocal    = numel(localMethods);
localDist = cell(nLocal, 1);
localRMSE = cell(nLocal, 1);

for di = 1:numel(allResults)
    D = allResults(di);
    for m = 1:nLocal
        mi = find(strcmp(D.methods, localMethods{m}));
        if isempty(mi); continue; end
        valid = isfinite(D.rmse(mi,:)) & abs(D.poseDistance) > 0.1;
        localDist{m} = [localDist{m} D.poseDistance(valid)];
        localRMSE{m} = [localRMSE{m} D.rmse(mi, valid)];
    end
end

% Pool FFT Method
fftDist = [];
fftRMSE = [];
for di = 1:numel(fftResults)
    F     = fftResults(di);
    valid = isfinite(F.rmse) & abs(F.poseDistance) > 0.1;
    fftDist = [fftDist F.poseDistance(valid)];
    fftRMSE = [fftRMSE F.rmse(valid)];
end

% Rank Local Methods Worst to Best
meanLocal = nan(1, nLocal);
for m = 1:nLocal
    if ~isempty(localRMSE{m})
        meanLocal(m) = mean(localRMSE{m}, 'omitnan');
    end
end
[~, order] = sort(meanLocal, 'descend');

% Print Summary
fprintf('=== Summary (pooled) ===\n');
fprintf('%-15s  %8s  %8s\n', 'Method', 'MeanRMSE', 'N points');
for k = 1:nLocal
    m = order(k);
    fprintf('%-15s  %8.3f  %8d\n', localMethods{m}, meanLocal(m), numel(localRMSE{m}));
end
if ~isempty(fftRMSE)
    fprintf('%-15s  %8.3f  %8d\n', 'FFTLookup', mean(fftRMSE, 'omitnan'), numel(fftRMSE));
end

% Plot
colors = lines(nLocal + 1);
figure(1); clf; hold on;

[sd, si] = sort(fftDist);
plot(sd, fftRMSE(si), 'Color', 'k', 'LineWidth', 1.5, 'DisplayName', 'FFTLookup');

for k = 1:nLocal
    m = order(k);
    if isempty(localDist{m}); continue; end
    [sd, si]  = sort(localDist{m});
    sr        = localRMSE{m}(si);
    windowSize = round(numel(sr) * 0.005);
    % sr_smooth  = movmean(sr, windowSize); % Can smooth if needed for larger number of datasets
    plot(sd, sr, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', localMethods{m});
end

xline(0, '--k', 'HandleVisibility', 'off');
xlabel('Pose Distance (m)', 'FontSize', 12);
ylabel('RMSE (m)', 'FontSize', 12);
title('RMSE vs Pose Distance — Smoothed', 'FontSize', 13);
legend('FFT Lookup', 'Nearest Neighbor', 'Linear', 'Bilinear Splat', ...
    'Bilinear Splat with Gaussian Convolution', 'Location', 'northeast', 'FontSize', 15);
grid on;
set(gcf, 'Position', [100 100 900 500]);