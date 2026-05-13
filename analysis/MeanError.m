% MeanError.m
% Computes RMSE vs pose distance for a single dataset across all frames
% Runs Splat, NN, Linear, and Natural neighbor methods
clear; clc;

% Dataset config
prm.refFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_450.mat';
prm.refFrameNum = 30;

% Sensor parameters
prm.interpMethod = 1;
prm.azStart = 2*pi;
prm.azDir   = -1;
prm.minEl   = -11.27*pi/180;
prm.maxEl   =  10.63*pi/180;

% Load reference
fprintf("Reference file: %s\n", prm.refFileName)
fprintf("Reference frame: %d\n", prm.refFrameNum)
load(prm.refFileName);
refPC   = pc{prm.refFrameNum};
prm.nEl = size(refPC, 1);
prm.nAz = size(refPC, 2);
prm     = generateMask(prm);

refXYZ = pc2xyz(refPC);
refAER = vec_xyz2aer(refXYZ, prm);
refR   = reshape(refAER(3,:), prm.nEl, prm.nAz);
refR(~isfinite(refR) | refR <= 0) = NaN;

az = linspace(prm.azStart, prm.azStart + prm.azDir*2*pi, prm.nAz);
el = linspace(prm.maxEl, prm.minEl, prm.nEl);
[AzGrid, ElGrid] = meshgrid(az, el);

% Preallocate
frameNums = 1:60;
nFrames   = numel(frameNums);

meanErrorSplat   = nan(1, nFrames);
meanErrorNN      = nan(1, nFrames);
meanErrorLinear  = nan(1, nFrames);
meanErrorNat     = nan(1, nFrames);
stdSplat         = nan(1, nFrames);
stdNN            = nan(1, nFrames);
stdLinear        = nan(1, nFrames);
stdNat           = nan(1, nFrames);
frameOffset      = zeros(1, nFrames);
distance         = zeros(1, nFrames);
rotation         = zeros(1, nFrames);
poseDistance     = zeros(1, nFrames);
nValidSplat      = zeros(1, nFrames);
nValidNN         = zeros(1, nFrames);
nValidLinear     = zeros(1, nFrames);
nValidNat        = zeros(1, nFrames);

% Main loop
for fi = 1:nFrames
    fNum          = frameNums(fi);
    frameOffset(fi) = fNum - prm.refFrameNum;

    newPC  = pc{fNum};
    newXYZ = pc2xyz(newPC);

    % ICP registration
    pcNew     = pointCloud(newXYZ');
    pcRef     = pointCloud(refXYZ');
    tform     = pcregistericp(pcNew, pcRef, Metric="planeToPlane");
    prm.R     = tform.Rotation;
    prm.delta = tform.Translation';

    distance(fi)     = sign(frameOffset(fi)) * norm(prm.delta);
    rotation(fi)     = acos((trace(prm.R) - 1) / 2);
    d                = 25;  % rotation weight (meters per radian)
    poseDistance(fi) = distance(fi) + d * rotation(fi);

    % Transform new frame into reference coordinates
    newXYZref = prm.R * double(newXYZ) + prm.delta;
    newAERref = vec_xyz2aer(newXYZref, prm);

    % Run methods
    newAERsplat = AERnurecon(newAERref, prm);
    splatRange  = reshape(newAERsplat(3,:), prm.nEl, prm.nAz);
    splatRange(~isfinite(splatRange) | splatRange <= 0) = NaN;

    azNew = newAERref(1,:)';
    elNew = newAERref(2,:)';
    rNew  = newAERref(3,:)';
    valid = isfinite(rNew) & rNew > 0;

    F_nn  = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'nearest', 'none');
    F_lin = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'linear',  'none');
    F_nat = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'natural', 'none');

    nnRange  = F_nn( AzGrid, ElGrid);
    linRange = F_lin(AzGrid, ElGrid);
    natRange = F_nat(AzGrid, ElGrid);
    nnRange( ~isfinite(nnRange)  | nnRange  <= 0) = NaN;
    linRange(~isfinite(linRange) | linRange <= 0) = NaN;
    natRange(~isfinite(natRange) | natRange <= 0) = NaN;

    % Valid pixel counts
    refValid     = isfinite(refR);
    validSplat   = refValid & isfinite(splatRange);
    validNN      = refValid & isfinite(nnRange);
    validLinear  = refValid & isfinite(linRange);
    validNat     = refValid & isfinite(natRange);

    nValidSplat(fi)  = sum(validSplat(:));
    nValidNN(fi)     = sum(validNN(:));
    nValidLinear(fi) = sum(validLinear(:));
    nValidNat(fi)    = sum(validNat(:));

    % RMSE and std per frame
    errorSplat  = (splatRange(validSplat)  - refR(validSplat)).^2;
    errorNN     = (nnRange(validNN)        - refR(validNN)).^2;
    errorLinear = (linRange(validLinear)   - refR(validLinear)).^2;
    errorNat    = (natRange(validNat)      - refR(validNat)).^2;

    meanErrorSplat(fi)  = sqrt(mean(errorSplat));
    meanErrorNN(fi)     = sqrt(mean(errorNN));
    meanErrorLinear(fi) = sqrt(mean(errorLinear));
    meanErrorNat(fi)    = sqrt(mean(errorNat));

    stdSplat(fi)  = std(errorSplat);
    stdNN(fi)     = std(errorNN);
    stdLinear(fi) = std(errorLinear);
    stdNat(fi)    = std(errorNat);

    fprintf('Frame %d (offset %+d): dist=%.1fm  splat=%d  linear=%d valid px\n', ...
        fNum, frameOffset(fi), distance(fi), nValidSplat(fi), nValidLinear(fi));
end

% Plot
figure; clf;
plot(poseDistance, meanErrorSplat,  'b', 'DisplayName', 'Splat'); hold on;
plot(poseDistance, meanErrorNN,     'r', 'DisplayName', 'NN');
plot(poseDistance, meanErrorLinear, 'g', 'DisplayName', 'Linear');
plot(poseDistance, meanErrorNat,    'm', 'DisplayName', 'Natural');
ylabel('RMSE (m)');
xlabel('Pose Distance from Reference (m)');
title('RMSE vs Pose Distance');
legend('Location', 'best'); grid on;
xline(0);

% Summary
fprintf('\nMean errors across all frames (omitting NaN):\n');
fprintf('  Splat:   %.4f m  (mean valid px: %.0f)\n', mean(meanErrorSplat,  'omitnan'), mean(nValidSplat));
fprintf('  Nearest: %.4f m  (mean valid px: %.0f)\n', mean(meanErrorNN,     'omitnan'), mean(nValidNN));
fprintf('  Linear:  %.4f m  (mean valid px: %.0f)\n', mean(meanErrorLinear, 'omitnan'), mean(nValidLinear));
fprintf('  Natural: %.4f m  (mean valid px: %.0f)\n', mean(meanErrorNat,    'omitnan'), mean(nValidNat));


%% Functions

% Converts [3xN] XYZ to [3xN] AER with azimuth unwrapping
% In: listXYZ [3xN], prm  Out: listAER [3xN]
function listAER = vec_xyz2aer(listXYZ, prm)
    AzRaw      = atan2(listXYZ(2,:), listXYZ(1,:));
    Az         = azUnwrap(AzRaw, prm);
    horizRange = sqrt(listXYZ(1,:).^2 + listXYZ(2,:).^2);
    El         = atan2(listXYZ(3,:), horizRange);
    range      = sqrt(listXYZ(3,:).^2 + horizRange.^2);
    listAER    = [Az; El; range];
end

% Splats points onto oversampled grid with bilinear weights, fills holes via Gaussian spectral smoothing
% In: listAER [3xN], prm  Out: newAERinterp [3 x nEl*nAz]
function newAERinterp = AERnurecon(listAER, prm)
    upsample = 2;
    nElDec   = prm.nEl * upsample;
    nAzDec   = prm.nAz * upsample;
    elTicks  = linspace(prm.maxEl, prm.minEl, nElDec);
    azTicks  = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;
    curR  = listAER(3,:);
    valid = curR > 0 & isfinite(curR) & curR < 200;
    az    = listAER(1, valid);
    el    = listAER(2, valid);
    curR  = curR(valid);
    aPos = (az - azTicks(1)) / (azTicks(2) - azTicks(1)) + 1;
    ePos = (el - elTicks(1)) / (elTicks(2) - elTicks(1)) + 1;
    a0 = floor(aPos); a1 = a0+1; e0 = floor(ePos); e1 = e0+1;
    wa1 = aPos-a0; wa0 = 1-wa1; we1 = ePos-e0; we0 = 1-we1;
    a0 = max(min(a0,nAzDec),1); a1 = max(min(a1,nAzDec),1);
    e0 = max(min(e0,nElDec),1); e1 = max(min(e1,nElDec),1);
    idx = [sub2ind([nElDec nAzDec],e0,a0); sub2ind([nElDec nAzDec],e0,a1);
           sub2ind([nElDec nAzDec],e1,a0); sub2ind([nElDec nAzDec],e1,a1)];
    w   = [we0.*wa0; we0.*wa1; we1.*wa0; we1.*wa1];
    wr4 = w .* repmat(curR(:)', 4, 1);
    Rinterp = reshape(accumarray(idx(:), wr4(:), [nElDec*nAzDec 1]), nElDec, nAzDec);
    Winterp = reshape(accumarray(idx(:), w(:),   [nElDec*nAzDec 1]), nElDec, nAzDec);
    hasData = Winterp > 0;
    Rinterp(hasData) = Rinterp(hasData) ./ Winterp(hasData);
    M = double(hasData);
    [uaz, uel] = meshgrid(linspace(-0.5,0.5,nAzDec), linspace(-0.5,0.5,nElDec));
    Gwin  = ifftshift(exp(-(uaz.^2 + uel.^2) / (2*0.25^2)));
    LP_RM = real(ifft2(fft2(Rinterp .* M) .* Gwin));
    LP_M  = real(ifft2(fft2(M) .* Gwin));
    Rsmooth = zeros(nElDec, nAzDec);
    enough  = LP_M > 0.01;
    Rsmooth(enough) = LP_RM(enough) ./ LP_M(enough);
    Rinterp(~hasData &  enough) = Rsmooth(~hasData & enough);
    Rinterp(~hasData & ~enough) = 0;
    Rinterp(Rinterp < 0) = 0;
    rowIdx = round(linspace(1, nElDec, prm.nEl));
    colIdx = round(linspace(1, nAzDec, prm.nAz));
    Rout   = Rinterp(rowIdx, colIdx);
    Rout(Rout <= 0 | Rout > 200) = Inf;
    [AzGrid, ElGrid] = meshgrid(azTicks(colIdx), elTicks(rowIdx));
    nCols = prm.nEl * prm.nAz;
    newAERinterp = [reshape(AzGrid,1,nCols); reshape(ElGrid,1,nCols); reshape(Rout,1,nCols)];
end

% Builds ego-vehicle pixel mask, stores linear indices in prm.domainMaskColIndxAER
% In/Out: prm
% Note: mask coordinates determined manually for this vehicle and mount
function prm = generateMask(prm)
    maskRangeIm = false(prm.nEl, prm.nAz);
    maskRangeIm(127:128,:)         = true;
    maskRangeIm(118:128, 178:207)  = true;
    maskRangeIm(120:128, 818:842)  = true;
    maskRangeIm(115:128, 1203:1233)= true;
    maskRangeIm(115:128, 1838:1865)= true;
    maskRangeIm(120:128, 2036:2048)= true;
    maskRangeIm(120:128, 1:7)      = true;
    domainMask1D = reshape(maskRangeIm, [1 numel(maskRangeIm)]);
    prm.domainMaskColIndxAER = find(domainMask1D);
end

% Corrects azimuth wrap-around errors relative to nominal scan direction
% In: Az [nEl x nAz], prm  Out: correctedAz [nEl x nAz]
% Note: only corrects +-2pi slips
function correctedAz = azUnwrap(Az, prm)
    correctedAz = Az;
    numCols     = size(Az, 2);
    nomAngle    = linspace(prm.azStart, prm.azStart+prm.azDir*2*pi, numCols+1);
    nomAngle(end) = [];
    nomAngleMat   = repmat(nomAngle, size(Az,1), 1);
    angleDiff     = Az - nomAngleMat;
    correctedAz(angleDiff >  pi/2) = correctedAz(angleDiff >  pi/2) - 2*pi;
    correctedAz(angleDiff < -pi/2) = correctedAz(angleDiff < -pi/2) + 2*pi;
end

% Converts point cloud struct (nEl x nAz x 3) to [3xN] XYZ array
% In: pc  Out: vecXYZ [3xN]
function vecXYZ = pc2xyz(pc)
    totPix = size(pc,1) * size(pc,2);
    vecXYZ = [reshape(pc(:,:,1), [1 totPix]);
              reshape(pc(:,:,2), [1 totPix]);
              reshape(pc(:,:,3), [1 totPix])];
end