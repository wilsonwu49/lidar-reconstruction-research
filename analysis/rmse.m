% rmse.m
% Computes RMSE for all methods (Splat, NN, Linear, BilinearSplat, Triangle, FFTLookup) across frames
% Applies ego mask; results saved to benchmark_results.mat
clear; clc;

% Dataset config
cfg.fileName    = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';
cfg.refFrameNum = 30;
cfg.frameNums   = 1:60;
cfg.datasetTag  = 'Volpe_1500_ref30';

resultsFile = 'C:\Users\wilso\RifeResearch\benchmark_results.mat';

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
refAER(3, prm.domainMaskColIndxAER) = -1;

refR = reshape(refAER(3,:), prm.nEl, prm.nAz);
refR(~isfinite(refR) | refR <= 0 | refR > 200) = NaN;

refAERinterp = vec_aerInterp(refAER, prm);

az = linspace(prm.azStart, prm.azStart + prm.azDir*2*pi, prm.nAz);
el = linspace(prm.maxEl, prm.minEl, prm.nEl);
[AzGrid, ElGrid] = meshgrid(az, el);

% Methods
methods = {'Splat','NN','Linear','Bilinear Splat','Triangle','FFTLookup'};
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
    rotation(fi)     = acos(min(1, max(-1, (trace(prm.R) - 1) / 2)));
    poseDistance(fi) = distance(fi);

    % Transform into reference coordinates with ego mask
    newXYZref = prm.R * double(newXYZ) + prm.delta;
    newAERref = vec_xyz2aer(newXYZref, prm);
    newAERref(3, prm.domainMaskColIndxAER) = -1;

    % Raw new frame AER in its own coordinate system (needed for FFTLookup)
    newAER_raw = vec_xyz2aer(newXYZ, prm);

    azNew = newAERref(1,:)';
    elNew = newAERref(2,:)';
    rNew  = newAERref(3,:)';
    valid = isfinite(rNew) & rNew > 0 & rNew < 200;

    % Run methods
    rangeResults = cell(nMeth, 1);

    splatAER = AERnurecon(newAERref, prm);
    R_splat  = reshape(splatAER(3,:), prm.nEl, prm.nAz);
    R_splat(~isfinite(R_splat) | R_splat <= 0 | R_splat > 200) = NaN;
    rangeResults{1} = R_splat;

    F_nn  = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'nearest', 'none');
    F_lin = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'linear',  'none');
    for m = 2:3
        switch m
            case 2; R_m = F_nn( AzGrid, ElGrid);
            case 3; R_m = F_lin(AzGrid, ElGrid);
        end
        R_m(~isfinite(R_m) | R_m <= 0 | R_m > 200) = NaN;
        rangeResults{m} = R_m;
    end

    R_rawsplat = rawSplat(newAERref, prm);
    R_rawsplat(~isfinite(R_rawsplat) | R_rawsplat <= 0 | R_rawsplat > 200) = NaN;
    rangeResults{4} = R_rawsplat;

    prm.interpMethod = 1;
    triAER = vec_aerInterp(newAERref, prm);
    R_tri  = reshape(triAER(3,:), prm.nEl, prm.nAz);
    R_tri(~isfinite(R_tri) | R_tri <= 0 | R_tri > 200) = NaN;
    rangeResults{5} = R_tri;

    fftAER = fft2lookupreverse(newAER_raw, refAER, prm);
    R_fft  = reshape(fftAER(3,:), prm.nEl, prm.nAz);
    R_fft(~isfinite(R_fft) | R_fft <= 0 | R_fft > 200) = NaN;
    rangeResults{6} = R_fft;

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

    fprintf('  %-12s %8s %8s %8s %6s\n', 'Method','RMSE','Median','95pct','Cov%');
    for m = 1:nMeth
        fprintf('  %-12s %8.2f %8.2f %8.2f %5.1f%%\n', ...
            methods{m}, rmse_all(m,fi), median_all(m,fi), pct95_all(m,fi), coverage_all(m,fi));
    end

    % Visualization
    subplot(nMeth+1, 1, 1);
    vizAzEl(refAERinterp, prm);
    title('Reference');
    for m = 1:nMeth
        subplot(nMeth+1, 1, m+1);
        imagesc(rangeResults{m}); axis off;
        title(sprintf('%s  RMSE=%.2fm  med=%.2fm  cov=%.0f%%', ...
            methods{m}, rmse_all(m,fi), median_all(m,fi), coverage_all(m,fi)));
    end
    drawnow;
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
    fprintf('Overwrote existing entry: %s\n', cfg.datasetTag);
else
    allResults = [allResults newEntry];
    fprintf('Added new entry: %s\n', cfg.datasetTag);
end

save(resultsFile, 'allResults');
fprintf('Saved to %s  (%d total datasets)\n', resultsFile, numel(allResults));

% Plot
plotBenchmark(allResults, methods);


%% Functions

% Plots RMSE, median, 95th pct, and coverage vs pose distance for all stored datasets
% In: allResults struct array, methods cell array  Out: none (figure 20)
function plotBenchmark(allResults, methods)
    colors  = lines(numel(methods));
    markers = {'o','s','d','^','v','p','h'};
    metricNames  = {'rmse','median_err','pct95','coverage'};
    metricLabels = {'RMSE (m)','Median Error (m)','95th Percentile Error (m)','Coverage (%)'};
    nMetrics     = numel(metricNames);
    figure(20); clf;
    sgtitle('Benchmark Results — All Datasets', 'FontSize', 13, 'FontWeight', 'bold');
    for mi = 1:nMetrics
        subplot(nMetrics, 1, mi); hold on;
        for di = 1:numel(allResults)
            D   = allResults(di);
            dat = D.(metricNames{mi});
            for m = 1:numel(methods)
                plot(D.poseDistance, dat(m,:), ...
                     'Color', colors(m,:), 'Marker', markers{di}, ...
                     'LineStyle', '-', 'LineWidth', 1.2, ...
                     'DisplayName', sprintf('%s (%s)', methods{m}, D.tag));
            end
        end
        ylabel(metricLabels{mi});
        xlabel('Pose Distance (m)');
        xline(0, '--k', 'HandleVisibility','off');
        grid on;
        if mi == 1; legend('Location','best','FontSize',7); end
    end
end

% Synthesizes reference-frame view by evaluating the new frame FFT at fractional indices
% corresponding to where each reference pixel lands in the new frame coordinate system
% In: newAER [3xN] new frame in its own coords, refAER [3xN], prm
% Out: newAERinterp [3 x nEl*nAz]
function newAERinterp = fft2lookupreverse(newAER, refAER, prm)
    % Build FFT of new frame range image (mean-subtract to reduce DC bias)
    newR    = reshape(newAER(3,:), prm.nEl, prm.nAz);
    invalid = ~isfinite(newR) | newR <= 0;
    newMean = mean(newR(~invalid));
    newR(invalid) = newMean;
    F_new   = fft2(newR - newMean);
    [N, M]  = size(F_new);
    Fvec    = F_new(:);
    [Krow, Kcol] = ndgrid(0:N-1, 0:M-1);
    Krow = Krow(:); Kcol = Kcol(:);

    % Map reference pixels into new frame coordinates
    R_inv     = prm.R';
    delta_inv = -prm.R' * prm.delta;
    validRef  = isfinite(refAER(3,:)) & refAER(3,:) > 0 & refAER(3,:) < 200;
    validIdx  = find(validRef);
    refXYZ_valid = vec_aer2xyz(refAER(:, validRef));
    xyz_newframe = R_inv * refXYZ_valid + repmat(delta_inv, 1, size(refXYZ_valid,2));
    newFrameAER  = vec_xyz2aer(xyz_newframe, prm);

    % Fractional pixel indices in new frame
    AzN = newFrameAER(1,:);
    ElN = newFrameAER(2,:);
    u   = (prm.maxEl - ElN) ./ (prm.maxEl - prm.minEl) .* (N - 1);
    v   = (AzN - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
    u   = mod(u, N);
    v   = mod(v, M);

    % Chunked fractional IDFT: R(u,v) = (1/NM) * sum_k F_k * exp(j2pi*(k_row*u/N + k_col*v/M))
    nPts          = length(AzN);
    rangeNewFrame = zeros(1, nPts);
    CHUNK         = 1024;
    for i = 1:CHUNK:nPts
        idx   = i:min(i+CHUNK-1, nPts);
        phase = 2*pi * (Krow .* (u(idx)/N) + Kcol .* (v(idx)/M));
        rangeNewFrame(idx) = real(Fvec.' * exp(1j * phase)) / (N*M);
    end
    rangeNewFrame = rangeNewFrame + newMean;

    % Convert synthesized new-frame range back to reference-frame range
    synthAER  = [AzN; ElN; rangeNewFrame];
    xyz_new   = vec_aer2xyz(synthAER);
    xyz_ref   = prm.R * xyz_new + repmat(prm.delta, 1, nPts);
    rangeRef  = sqrt(sum(xyz_ref.^2, 1));
    rangeRef(rangeRef < 0.5 | rangeRef > 200) = Inf;

    rangeGrid           = Inf(N, M);
    rangeGrid(validIdx) = rangeRef;

    el_grid = linspace(prm.maxEl, prm.minEl, N);
    az_grid = prm.azStart + prm.azDir * (0:M-1) / M * 2*pi;
    [AzOut, ElOut] = meshgrid(az_grid, el_grid);
    nCols = N * M;
    newAERinterp = [reshape(AzOut,    1,nCols);
                    reshape(ElOut,    1,nCols);
                    reshape(rangeGrid,1,nCols)];
end

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

% Triangle-based forward-warp interpolation with face visibility check and isolated point snapping
% In: listAER [3xN], prm (interpMethod: 1=NN)
% Out: listAER [3 x nEl*nAz]
function listAER = vec_aerInterp(listAER, prm)
    [Amat,Emat,Rmat]=vec2mataer(listAER,[prm.nEl prm.nAz]);
    elTicks=linspace(prm.maxEl,prm.minEl,prm.nEl);
    azTicks=prm.azStart+prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    Rinterp=Inf(prm.nEl,prm.nAz);
    [isConsistSE,isConsistNW]=boxConsist(Amat,Emat);
    [rowSE,colSE]=find(isConsistSE); [rowNW,colNW]=find(isConsistNW);
    for n=1:length(rowSE)
        rn=rowSE(n); cn=colSE(n);
        aList=[Amat(rn+1,cn+1) Amat(rn+1,cn) Amat(rn,cn+1)];
        eList=[Emat(rn+1,cn+1) Emat(rn+1,cn) Emat(rn,cn+1)];
        rList=[Rmat(rn+1,cn+1) Rmat(rn+1,cn) Rmat(rn,cn+1)];
        Rinterp=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);
    end
    for n=1:length(rowNW)
        rn=rowNW(n); cn=colNW(n);
        aList=[Amat(rn,cn) Amat(rn+1,cn) Amat(rn,cn+1)];
        eList=[Emat(rn,cn) Emat(rn+1,cn) Emat(rn,cn+1)];
        rList=[Rmat(rn,cn) Rmat(rn+1,cn) Rmat(rn,cn+1)];
        Rinterp=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);
    end
    delAz=azTicks(1)-azTicks(2); delEl=elTicks(1)-elTicks(2);
    AzVecRnd=round((azTicks(1)-listAER(1,:))/delAz+1);
    ElVecRnd=round((elTicks(1)-listAER(2,:)+1e-7)/delEl+1);
    rVec=listAER(3,:);
    inRange=find(AzVecRnd>0&AzVecRnd<=prm.nAz&ElVecRnd>0&ElVecRnd<=prm.nEl&rVec>0&rVec<200);
    indxRnd=sub2ind([prm.nEl prm.nAz],ElVecRnd(inRange),AzVecRnd(inRange));
    replaceFlag=Rinterp(indxRnd)>rVec(inRange);
    Rinterp(indxRnd(replaceFlag))=rVec(inRange(replaceFlag));
    [Anew,Enew]=meshgrid(azTicks,elTicks);
    nCols=prm.nAz*prm.nEl;
    listAER=[reshape(Anew,1,nCols);reshape(Enew,1,nCols);reshape(Rinterp,1,nCols)];
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

% Converts [3xN] AER to [3xN] Cartesian XYZ
% In: listAER [3xN]  Out: listXYZ [3xN]
function listXYZ = vec_aer2xyz(listAER)
    az=listAER(1,:); el=listAER(2,:); r=listAER(3,:);
    listXYZ=[r.*cos(el).*cos(az);r.*cos(el).*sin(az);r.*sin(el)];
end

% Reshapes [3xN] AER vector into three (nEl x nAz) matrices for Az, El, R
% In: vecAER [3xN], imSize [nEl nAz]  Out: matA, matE, matR each [nEl x nAz]
function [matA,matE,matR] = vec2mataer(vecAER, imSize)
    matA=reshape(vecAER(1,:),imSize(1:2));
    matE=reshape(vecAER(2,:),imSize(1:2));
    matR=reshape(vecAER(3,:),imSize(1:2));
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

% Displays range image as grayscale (closer = brighter)
% In: imAER [3xN], prm  Out: none
function vizAzEl(imAER, prm)
    [~,~,Rmat]=vec2mataer(imAER,[prm.nEl size(imAER,2)/prm.nEl]);
    imagesc(3./Rmat); axis off;
end

% Fills a triangular face on the output grid using nearest-vertex range assignment
% In: aList, eList, rList [1x3] vertices; azTicks, elTicks; curView [nEl x nAz]
% Out: curView [nEl x nAz]
function curView = viewSynthesisNN(aList,eList,rList,azTicks,elTicks,curView)
    minA=min(aList)-1e-8; maxA=max(aList)+1e-8;
    minE=min(eList)-1e-8; maxE=max(eList)+1e-8;
    azTickInd=find(azTicks>=minA&azTicks<=maxA);
    elTickInd=find(elTicks>=minE&elTicks<=maxE);
    if isempty(azTickInd)||isempty(elTickInd); return; end
    [locA,locE]=meshgrid(azTicks(azTickInd),elTicks(elTickInd));
    v12=[aList(2)-aList(1) eList(2)-eList(1)];
    v23=[aList(3)-aList(2) eList(3)-eList(2)];
    v31=[aList(1)-aList(3) eList(1)-eList(3)];
    u12=v12/norm(v12); u23=v23/norm(v23); u31=v31/norm(v31);
    vert1=u12(1)*(locA-aList(1))+u12(2)*(locE-eList(1))<norm(v12)/2&-u31(1)*(locA-aList(1))-u31(2)*(locE-eList(1))<=norm(v31)/2;
    vert2=-u12(1)*(locA-aList(2))-u12(2)*(locE-eList(2))<=norm(v12)/2&u23(1)*(locA-aList(2))+u23(2)*(locE-eList(2))<norm(v23)/2;
    vert3=-u23(1)*(locA-aList(3))-u23(2)*(locE-eList(3))<=norm(v23)/2&u31(1)*(locA-aList(3))+u31(2)*(locE-eList(3))<norm(v31)/2;
    inMask=inTriangle(aList,eList,locA,locE);
    locR=Inf(size(locA));
    locR(inMask&vert1)=rList(1); locR(inMask&vert2)=rList(2); locR(inMask&vert3)=rList(3);
    blockR=curView(elTickInd,azTickInd);
    blockR(blockR>locR)=locR(blockR>locR);
    curView(elTickInd,azTickInd)=blockR;
end

% Checks face visibility after rigid transform using Az/El edge cross products
% In: aGrid, eGrid [nEl x nAz]  Out: isConsistSE, isConsistNW logical [(nEl-1) x (nAz-1)]
function [isConsistSE,isConsistNW] = boxConsist(aGrid,eGrid)
    dAlat=aGrid(:,1:end-1)-aGrid(:,2:end); dEvrt=eGrid(1:end-1,:)-eGrid(2:end,:);
    dElat=eGrid(:,1:end-1)-eGrid(:,2:end); dAvrt=aGrid(1:end-1,:)-aGrid(2:end,:);
    a=dAlat(2:end,:); b=dElat(2:end,:); c=dAvrt(:,2:end); d=dEvrt(:,2:end);
    isConsistSE=(a.*d-b.*c)>0;
    a=dAlat(1:end-1,:); b=dElat(1:end-1,:); c=dAvrt(:,1:end-1); d=dEvrt(:,1:end-1);
    isConsistNW=(a.*d-b.*c)>0;
end

% Tests whether grid points fall inside a triangle using the half-plane method
% In: vx, vy [1x3] vertices; xGrid, yGrid meshgrid arrays  Out: inMask logical
function inMask = inTriangle(vx,vy,xGrid,yGrid)
    vMat=[vx;vy;zeros(1,3)]; khat=[0 0 1];
    v12=vMat(:,2)-vMat(:,1); v23=vMat(:,3)-vMat(:,2); v31=vMat(:,1)-vMat(:,3);
    nv12=cross(v12,khat); nv23=cross(v23,khat); nv31=cross(v31,khat);
    inside12=(nv12(1)*(xGrid-vx(1))+nv12(2)*(yGrid-vy(1)))*sign(dot(nv12,-v31))>=-1e-5;
    inside23=(nv23(1)*(xGrid-vx(2))+nv23(2)*(yGrid-vy(2)))*sign(dot(nv23,-v12))>=-1e-5;
    inside31=(nv31(1)*(xGrid-vx(3))+nv31(2)*(yGrid-vy(3)))*sign(dot(nv31,-v23))>=-1e-5;
    inMask=inside12&inside23&inside31;
end