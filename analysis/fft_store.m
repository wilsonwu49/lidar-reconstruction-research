% fft_store.m
% Extracts FFT Lookup results from benchmark_results.mat into a separate fft_results.mat
clear; clc;

% File paths
resultsFile    = 'C:\Users\wilso\RifeResearch\benchmark_results.mat';
fftResultsFile = 'C:\Users\wilso\RifeResearch\fft_results.mat';

% Load and list datasets
load(resultsFile, 'allResults');
fprintf('Loaded %d datasets:\n', numel(allResults));
for di = 1:numel(allResults)
    fprintf('  %d. %s  methods: %s\n', di, allResults(di).tag, strjoin(allResults(di).methods, ', '));
end

% Extract FFT data from each dataset
fftResults = [];
for di = 1:numel(allResults)
    D  = allResults(di);
    mi = find(strcmp(D.methods, 'FFTLookup'), 1);
    if isempty(mi)
        fprintf('No FFTLookup found in dataset %s — skipping\n', D.tag);
        continue;
    end
    entry.tag          = D.tag;
    entry.frameNums    = D.frameNums;
    entry.poseDistance = D.poseDistance;
    entry.distance     = D.distance;
    entry.rotation     = D.rotation;
    entry.rmse         = D.rmse(mi,:);
    entry.median_err   = D.median_err(mi,:);
    entry.pct95        = D.pct95(mi,:);
    entry.coverage     = D.coverage(mi,:);
    fprintf('  Dataset %s: %d / %d frames have FFT data\n', ...
        D.tag, sum(isfinite(entry.rmse)), numel(D.frameNums));
    fftResults = [fftResults entry];
end

% Save and print summary
if isempty(fftResults)
    fprintf('No FFT data found in any dataset.\n');
else
    save(fftResultsFile, 'fftResults');
    fprintf('\nSaved FFT results to %s\n', fftResultsFile);
    fprintf('\n=== FFT Results Summary ===\n');
    for di = 1:numel(fftResults)
        F     = fftResults(di);
        valid = isfinite(F.rmse);
        fprintf('Dataset: %s\n',                       F.tag);
        fprintf('  Frames with data: %d\n',            sum(valid));
        fprintf('  Pose distance range: %.2f to %.2f m\n', min(F.poseDistance(valid)), max(F.poseDistance(valid)));
        fprintf('  Mean RMSE:   %.3f m\n',             mean(F.rmse(valid),       'omitnan'));
        fprintf('  Mean Median: %.3f m\n',             mean(F.median_err(valid), 'omitnan'));
        fprintf('  Mean 95pct:  %.3f m\n',             mean(F.pct95(valid),      'omitnan'));
    end
end