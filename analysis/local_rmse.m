% local_rmse.m
% Computes RMSE for local methods (Splat, NN, Linear, BilinearSplat) across frames
% No ego mask applied; results saved to benchmark_results1.mat
clear; clc;

% Dataset config
cfg.fileName    = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_2200.mat';
cfg.refFrameNum = 30;
cfg.frameNums   = 1:60;
cfg.datasetTag  = 'Volpe_2200_ref30_v2';

resultsFile = 'C:\Users\wilso\RifeResearch\benchmark_results1.mat';

% Sensor parameters
prm.azStart      = 2*pi;
prm.azDir        = -1;
prm.minEl        = -11.27*pi/180;
prm.maxEl        =  10.63*pi/180;
prm.interpMethod = 1;

% Load reference
fprintf('Loading %s, reference frame %d\n', cfg.fileName, cfg.refFrameNum);
load(cfg.fileName);

refPC   = pc{cfg.refFrameNum};
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

% Methods
methods = {'Splat','NN','Linear','BilinearSplat'};
nMeth   = numel(methods);

% Preallocate
nFrames      = numel(cfg.frameNums);
poseDistance = zeros(1, nFrames);
distance     = zeros(1, nFrames);
rotation     = zeros(1, nFrames);

rmse_all     = nan(nMeth, nFrames);
median_all   = nan(nMeth, nFrames);
pct95_all    = nan(nMeth, nFrames);
coverage_all = nan(nMeth, nFrames);

% Main loop
for fi = 1:nFrames
    fNum = cfg.frameNums(fi);
    fprintf('\nFrame %d / %d (frame num %d) ...\n', fi, nFrames, fNum);

    newPC  = pc{fNum};
    newXYZ = pc2xyz(newPC);

    % ICP registration
    pcNew     = pointCloud(newXYZ');
    pcRef     = pointCloud(refXYZ');
    tform     = pcregistericp(pcNew, pcRef, Metric="planeToPlane");
    prm.R     = tform.Rotation;
    prm.delta = tform.Translation';

    frameOffset      = fNum - cfg.refFrameNum;
    distance(fi)     = sign(frameOffset) * norm(prm.delta);
    rotation(fi)     = acos(min(1, max(-1, (trace(prm.R)-1)/2)));
    poseDistance(fi) = distance(fi);

    % Transform into reference coordinates (no ego mask)
    newXYZref = prm.R * double(newXYZ) + prm.delta;
    newAERref = vec_xyz2aer(newXYZref, prm);

    azNew = newAERref(1,:)';
    elNew = newAERref(2,:)';
    rNew  = newAERref(3,:)';
    valid = isfinite(rNew) & rNew > 0 & rNew < 200;

    % Run methods
    rangeResults = cell(nMeth, 1);

    R_splat = AERnurecon(newAERref, prm);
    R_splat = reshape(R_splat(3,:), prm.nEl, prm.nAz);
    R_splat(~isfinite(R_splat) | R_splat <= 0 | R_splat > 200) = NaN;
    rangeResults{1} = R_splat;

    F_nn  = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'nearest', 'none');
    R_nn  = F_nn(AzGrid, ElGrid);
    R_nn(~isfinite(R_nn) | R_nn <= 0 | R_nn > 200) = NaN;
    rangeResults{2} = R_nn;

    F_lin = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'linear', 'none');
    R_lin = F_lin(AzGrid, ElGrid);
    R_lin(~isfinite(R_lin) | R_lin <= 0 | R_lin > 200) = NaN;
    rangeResults{3} = R_lin;

    R_raw = rawSplat(newAERref, prm);
    R_raw(~isfinite(R_raw) | R_raw <= 0 | R_raw > 200) = NaN;
    rangeResults{4} = R_raw;

    % Compute metrics
    refValid   = isfinite(refR);
    totalValid = sum(refValid(:));

    for m = 1:nMeth
        R_m   = rangeResults{m};
        vmask = refValid & isfinite(R_m);
        nv    = sum(vmask(:));
        coverage_all(m, fi) = nv / totalValid * 100;
        if nv > 0
            err_abs = abs(R_m(vmask) - refR(vmask));
            rmse_all(m,   fi) = sqrt(mean(err_abs.^2));
            median_all(m, fi) = median(err_abs);
            pct95_all(m,  fi) = quantile(err_abs, 0.95);
        end
    end

    fprintf('  %-15s %8s %8s %8s %6s\n', 'Method','RMSE','Median','95pct','Cov%');
    for m = 1:nMeth
        fprintf('  %-15s %8.2f %8.2f %8.2f %5.1f%%\n', ...
            methods{m}, rmse_all(m,fi), median_all(m,fi), pct95_all(m,fi), coverage_all(m,fi));
    end
end

% Save
newEntry.tag          = cfg.datasetTag;
newEntry.fileName     = cfg.fileName;
newEntry.refFrameNum  = cfg.refFrameNum;
newEntry.frameNums    = cfg.frameNums;
newEntry.poseDistance = poseDistance;
newEntry.distance     = distance;
newEntry.rotation     = rotation;
newEntry.methods      = methods;
newEntry.rmse         = rmse_all;
newEntry.median_err   = median_all;
newEntry.pct95        = pct95_all;
newEntry.coverage     = coverage_all;

if isfile(resultsFile)
    load(resultsFile, 'allResults');
else
    allResults = [];
end

existingTags = {};
for i = 1:numel(allResults)
    existingTags{i} = allResults(i).tag;
end
tagIdx = find(strcmp(existingTags, cfg.datasetTag));
if ~isempty(tagIdx)
    allResults(tagIdx) = newEntry;
    fprintf('\nOverwrote existing entry: %s\n', cfg.datasetTag);
else
    allResults = [allResults newEntry];
    fprintf('\nAdded new entry: %s\n', cfg.datasetTag);
end

save(resultsFile, 'allResults');
fprintf('Saved to %s  (%d total datasets)\n', resultsFile, numel(allResults));

% Plot
colors = lines(nMeth);
figure(1); clf; hold on;
for m = 1:nMeth
    valid = isfinite(rmse_all(m,:)) & abs(poseDistance) > 0.1;
    [sd, si] = sort(poseDistance(valid));
    sr = rmse_all(m, valid);
    plot(sd, sr(si), 'Color', colors(m,:), 'LineWidth', 1.5, 'DisplayName', methods{m});
end
xline(0, '--k', 'HandleVisibility', 'off');
xlabel('Pose Distance (m)', 'FontSize', 12);
ylabel('RMSE (m)', 'FontSize', 12);
title(sprintf('RMSE vs Pose Distance — %s', cfg.datasetTag), 'FontSize', 13);
legend('Location', 'best', 'FontSize', 9);
grid on;
set(gcf, 'Position', [100 100 900 500]);


%% Functions

% Splats points onto oversampled grid with bilinear weights, fills holes via Gaussian spectral smoothing
% In: listAER [3xN], prm  Out: newAERinterp [3 x nEl*nAz]
function newAERinterp = AERnurecon(listAER, prm)
    upsample = 2;
    nElDec = prm.nEl * upsample;
    nAzDec = prm.nAz * upsample;
    elTicks = linspace(prm.maxEl, prm.minEl, nElDec);
    azTicks = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;
    curR  = listAER(3,:);
    valid = curR > 0 & isfinite(curR) & curR < 200;
    az    = listAER(1,valid);
    el    = listAER(2,valid);
    curR  = curR(valid);
    aPos=(az-azTicks(1))/(azTicks(2)-azTicks(1))+1;
    ePos=(el-elTicks(1))/(elTicks(2)-elTicks(1))+1;
    a0=floor(aPos); a1=a0+1; e0=floor(ePos); e1=e0+1;
    wa1=aPos-a0; wa0=1-wa1; we1=ePos-e0; we0=1-we1;
    a0=max(min(a0,nAzDec),1); a1=max(min(a1,nAzDec),1);
    e0=max(min(e0,nElDec),1); e1=max(min(e1,nElDec),1);
    idx=[sub2ind([nElDec nAzDec],e0,a0);sub2ind([nElDec nAzDec],e0,a1);
         sub2ind([nElDec nAzDec],e1,a0);sub2ind([nElDec nAzDec],e1,a1)];
    w=[we0.*wa0;we0.*wa1;we1.*wa0;we1.*wa1];
    wr4=w.*repmat(curR(:)',4,1);
    Rinterp=reshape(accumarray(idx(:),wr4(:),[nElDec*nAzDec 1]),nElDec,nAzDec);
    Winterp=reshape(accumarray(idx(:),w(:),  [nElDec*nAzDec 1]),nElDec,nAzDec);
    hasData=Winterp>0;
    Rinterp(hasData)=Rinterp(hasData)./Winterp(hasData);
    M_mask=double(hasData);
    [uaz,uel]=meshgrid(linspace(-0.5,0.5,nAzDec),linspace(-0.5,0.5,nElDec));
    Gwin=ifftshift(exp(-(uaz.^2+uel.^2)/(2*0.25^2)));
    LP_RM=real(ifft2(fft2(Rinterp.*M_mask).*Gwin));
    LP_M=real(ifft2(fft2(M_mask).*Gwin));
    Rsmooth=zeros(nElDec,nAzDec);
    enough=LP_M>0.01;
    Rsmooth(enough)=LP_RM(enough)./LP_M(enough);
    Rinterp(~hasData&enough)=Rsmooth(~hasData&enough);
    Rinterp(~hasData&~enough)=0;
    Rinterp(Rinterp<0)=0;
    rowIdx=round(linspace(1,nElDec,prm.nEl));
    colIdx=round(linspace(1,nAzDec,prm.nAz));
    Rout=Rinterp(rowIdx,colIdx);
    Rout(Rout<=0|Rout>200)=Inf;
    [AzGrid,ElGrid]=meshgrid(azTicks(colIdx),elTicks(rowIdx));
    nCols=prm.nEl*prm.nAz;
    newAERinterp=[reshape(AzGrid,1,nCols);reshape(ElGrid,1,nCols);reshape(Rout,1,nCols)];
end

% Bilinear splat only, no hole filling
% In: listAER [3xN], prm  Out: Rout [nEl x nAz]
function Rout = rawSplat(listAER, prm)
    upsample=2; nElDec=prm.nEl*upsample; nAzDec=prm.nAz*upsample;
    elTicks=linspace(prm.maxEl,prm.minEl,nElDec);
    azTicks=prm.azStart+prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;
    curR=listAER(3,:); valid=curR>0&isfinite(curR)&curR<200;
    az=listAER(1,valid); el=listAER(2,valid); curR=curR(valid);
    aPos=(az-azTicks(1))/(azTicks(2)-azTicks(1))+1;
    ePos=(el-elTicks(1))/(elTicks(2)-elTicks(1))+1;
    a0=floor(aPos); a1=a0+1; e0=floor(ePos); e1=e0+1;
    wa1=aPos-a0; wa0=1-wa1; we1=ePos-e0; we0=1-we1;
    a0=max(min(a0,nAzDec),1); a1=max(min(a1,nAzDec),1);
    e0=max(min(e0,nElDec),1); e1=max(min(e1,nElDec),1);
    idx=[sub2ind([nElDec nAzDec],e0,a0);sub2ind([nElDec nAzDec],e0,a1);
         sub2ind([nElDec nAzDec],e1,a0);sub2ind([nElDec nAzDec],e1,a1)];
    w=[we0.*wa0;we0.*wa1;we1.*wa0;we1.*wa1];
    wr4=w.*repmat(curR(:)',4,1);
    Rinterp=reshape(accumarray(idx(:),wr4(:),[nElDec*nAzDec 1]),nElDec,nAzDec);
    Winterp=reshape(accumarray(idx(:),w(:),  [nElDec*nAzDec 1]),nElDec,nAzDec);
    hasData=Winterp>0;
    Rinterp(hasData)=Rinterp(hasData)./Winterp(hasData);
    Rinterp(~hasData)=Inf;
    rowIdx=round(linspace(1,nElDec,prm.nEl));
    colIdx=round(linspace(1,nAzDec,prm.nAz));
    Rout=Rinterp(rowIdx,colIdx);
    Rout(Rout<=0|Rout>200)=Inf;
end

% Converts [3xN] XYZ to [3xN] AER with azimuth unwrapping
% In: listXYZ [3xN], prm  Out: listAER [3xN]
function listAER = vec_xyz2aer(listXYZ, prm)
    AzRaw=atan2(listXYZ(2,:),listXYZ(1,:));
    Az=azUnwrap(AzRaw,prm);
    horizRange=sqrt(listXYZ(1,:).^2+listXYZ(2,:).^2);
    El=atan2(listXYZ(3,:),horizRange);
    range=sqrt(listXYZ(3,:).^2+horizRange.^2);
    listAER=[Az;El;range];
end

% Corrects azimuth wrap-around errors relative to nominal scan direction
% In: Az [nEl x nAz], prm  Out: correctedAz [nEl x nAz]
% Note: only corrects +-2pi slips
function correctedAz = azUnwrap(Az, prm)
    correctedAz=Az;
    numCols=size(Az,2);
    nomAngle=linspace(prm.azStart,prm.azStart+prm.azDir*2*pi,numCols+1);
    nomAngle(end)=[];
    nomAngleMat=repmat(nomAngle,size(Az,1),1);
    angleDiff=Az-nomAngleMat;
    correctedAz(angleDiff>pi/2)=correctedAz(angleDiff>pi/2)-2*pi;
    correctedAz(angleDiff<-pi/2)=correctedAz(angleDiff<-pi/2)+2*pi;
end

% Builds ego-vehicle pixel mask, stores linear indices in prm.domainMaskColIndxAER
% In/Out: prm
% Note: mask coordinates determined manually for this vehicle and mount
function prm = generateMask(prm)
    maskRangeIm=false(prm.nEl,prm.nAz);
    maskRangeIm(127:128,:)=true;
    maskRangeIm(118:128,178:207)=true;
    maskRangeIm(120:128,818:842)=true;
    maskRangeIm(115:128,1203:1233)=true;
    maskRangeIm(115:128,1838:1865)=true;
    maskRangeIm(120:128,2036:2048)=true;
    maskRangeIm(120:128,1:7)=true;
    domainMask1D=reshape(maskRangeIm,[1 numel(maskRangeIm)]);
    prm.domainMaskColIndxAER=find(domainMask1D);
end

% Converts point cloud struct (nEl x nAz x 3) to [3xN] XYZ array
% In: pc  Out: vecXYZ [3xN]
function vecXYZ = pc2xyz(pc)
    totPix=size(pc,1)*size(pc,2);
    vecXYZ=[reshape(pc(:,:,1),[1 totPix]);
            reshape(pc(:,:,2),[1 totPix]);
            reshape(pc(:,:,3),[1 totPix])];
end