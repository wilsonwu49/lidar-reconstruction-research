prm.refFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_450.mat';
% prm.refFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1300.mat';
prm.refFrameNum = 30;
fprintf("Reference file: %s\n", prm.refFileName)
fprintf("Reference frame: %d\n", prm.refFrameNum)
load(prm.refFileName);
refPC = pc{prm.refFrameNum};
prm.interpMethod = 1;            % Options: 0 is no interpolation, 1 is nearest neighobr, 2 is planar fit


prm.nEl = size(refPC, 1);
prm.nAz = size(refPC, 2);
prm.azStart = 2*pi;
prm.azDir = -1;
prm.minEl = -11.27*pi/180;
prm.maxEl =  10.63*pi/180;
prm = generateMask(prm);

refXYZ = pc2xyz(refPC);
refAER = vec_xyz2aer(refXYZ, prm);
refR = reshape(refAER(3,:), prm.nEl, prm.nAz);
refR(~isfinite(refR) | refR <= 0) = NaN;

az = linspace(prm.azStart, prm.azStart + prm.azDir*2*pi, prm.nAz);
el = linspace(prm.maxEl, prm.minEl, prm.nEl);
[AzGrid, ElGrid] = meshgrid(az, el);

frameNums = [1:60];
nFrames = numel(frameNums);

meanErrorSplat = nan(1, nFrames);
meanErrorNN = nan(1, nFrames);
meanErrorLinear = nan(1, nFrames);
meanErrorNat = nan(1, nFrames);
stdSplat = nan(1, nFrames);
stdNN = nan(1, nFrames);
stdLinear = nan(1, nFrames);
stdNat = nan(1, nFrames);
frameOffset = zeros(1, nFrames);
distance = zeros(1, nFrames);
rotation = zeros(1, nFrames);
poseDistance = zeros(1,nFrames);
nValidSplat = zeros(1, nFrames);
nValidNN = zeros(1, nFrames);
nValidLinear = zeros(1, nFrames);
nValidNat = zeros(1, nFrames);

for fi = 1:nFrames
    fNum = frameNums(fi);
    frameOffset(fi) = fNum - prm.refFrameNum;

    newPC = pc{fNum};
    newXYZ = pc2xyz(newPC);

    pcNew = pointCloud(newXYZ');
    pcRef = pointCloud(refXYZ');
    tform = pcregistericp(pcNew, pcRef, Metric="planeToPlane");
    prm.R = tform.Rotation;
    prm.delta = tform.Translation';
    distance(fi) = sign(frameOffset(fi))*sqrt(prm.delta(1)^2 + prm.delta(2)^2 + prm.delta(3)^2);
    rotation(fi) = acos((trace(prm.R) - 1)/2);
    d = 25;
    poseDistance(fi) = distance(fi) + d * rotation(fi);

    newXYZref = prm.R * double(newXYZ) + prm.delta;
    newAERref = vec_xyz2aer(newXYZref, prm);

    newAERsplat = AERnurecon(newAERref, prm);
    splatRange = reshape(newAERsplat(3,:), prm.nEl, prm.nAz);
    splatRange(~isfinite(splatRange) | splatRange <= 0) = NaN;

    azNew = newAERref(1,:)';
    elNew = newAERref(2,:)';
    rNew = newAERref(3,:)';
    valid = isfinite(rNew) & rNew > 0;

    F_nn = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'nearest', 'none');
    F_lin = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'linear',  'none');
    F_nat = scatteredInterpolant(azNew(valid), elNew(valid), rNew(valid), 'natural', 'none');

    nnRange = F_nn(AzGrid,  ElGrid);
    linRange = F_lin(AzGrid, ElGrid);
    natRange = F_nat(AzGrid, ElGrid);
    nnRange(~isfinite(nnRange) | nnRange  <= 0) = NaN;
    linRange(~isfinite(linRange) | linRange <= 0) = NaN;
    natRange(~isfinite(natRange) | natRange <= 0) = NaN;

    % Checking Valid Points
    refValid = isfinite(refR);
    validSplat = refValid & isfinite(splatRange);
    validNN = refValid & isfinite(nnRange);
    validLinear = refValid & isfinite(linRange);
    validNat = refValid & isfinite(natRange);

    nValidSplat(fi) = sum(validSplat(:));
    nValidNN(fi) = sum(validNN(:));
    nValidLinear(fi) = sum(validLinear(:));
    nValidNat(fi) = sum(validNat(:));

    % Mean and STDev Error of Frame
    errorSplat = (splatRange(validSplat) - refR(validSplat)).^2;
    errorNN = (nnRange(validNN) - refR(validNN)).^2;
    errorLinear = (linRange(validLinear) - refR(validLinear)).^2;
    errorNat = (natRange(validNat) - refR(validNat)).^2;

    meanErrorSplat(fi) = sqrt(mean(errorSplat));
    meanErrorNN(fi) = sqrt(mean(errorNN));
    meanErrorLinear(fi) = sqrt(mean(errorLinear));
    meanErrorNat(fi) = sqrt(mean(errorNat));

    stdSplat(fi) = std(errorSplat);
    stdNN(fi) = std(errorNN);
    stdLinear(fi) = std(errorLinear);
    stdNat(fi) = std(errorNat);

    fprintf('Frame %d (offset %+d): distance =%d splat=%d built-in=%d valid px\n', fNum, frameOffset(fi), distance(fi), nValidSplat(fi), nValidLinear(fi));
end

figure; clf;
plot(poseDistance, meanErrorSplat,  'b', 'DisplayName', 'Splat'); hold on;
plot(poseDistance, meanErrorNN,     'r', 'DisplayName', 'NN');
plot(poseDistance, meanErrorLinear, 'g', 'DisplayName', 'Linear');
plot(poseDistance, meanErrorNat, 'm', 'DisplayName', 'Natural');
ylabel('Root Mean Square Error (m)');
xlabel('Translational Distance from Reference (m)');
title('RMSE vs Distance from Reference Frame')
legend('Location', 'best'); grid on;
xline(0);

% figure; clf;
% errorbar(poseDistance, meanErrorSplat,  stdSplat,  'b-o', 'DisplayName', 'Splat'); hold on;
% errorbar(poseDistance, meanErrorNN,     stdNN,     'r-o', 'DisplayName', 'NN');
% errorbar(poseDistance, meanErrorLinear, stdLinear, 'g-o', 'DisplayName', 'Linear');
% ylabel('Mean absolute range error (m)');
% xlabel('Distance from reference (frame 30)');
% title('Error vs Distance of Perspective');
% legend('Location', 'best'); grid on;
% xline(0);

% figure; clf;
% plot(frameOffset, nValidSplat, 'b', 'DisplayName', 'N valid (splat)'); hold on;
% plot(frameOffset, nValidNN, 'r', 'DisplayName', 'N valid (NN)');
% plot(frameOffset, nValidLinear, 'g', 'DisplayName', 'N valid (linear)');
% ylabel('Valid pixel count');
% xlabel('Frame offset from reference (frame 30)');
% title('Valid pixel count vs frame offset');
% legend('Location', 'best'); grid on;
% xline(0);
% 
% figure; clf;
% plot(poseDistance, nValidSplat, 'b', 'DisplayName', 'N valid (splat)'); hold on;
% plot(poseDistance, nValidNN, 'r', 'DisplayName', 'N valid (NN)');
% plot(poseDistance, nValidLinear, 'g', 'DisplayName', 'N valid (linear)');
% ylabel('Valid pixel count');
% xlabel('Distance from reference (frame 30)');
% title('Valid pixel count vs Distance of Perspective');
% legend('Location', 'best'); grid on;
% xline(0);

fprintf('\nMean errors across all frames (omitting NaN):\n');
fprintf('  Splat:   %.4f m  (mean valid px: %.0f)\n', mean(meanErrorSplat, 'omitnan'), mean(nValidSplat));
fprintf('  Nearest: %.4f m  (mean valid px: %.0f)\n', mean(meanErrorNN, 'omitnan'), mean(nValidNN));
fprintf('  Linear:  %.4f m  (mean valid px: %.0f)\n', mean(meanErrorLinear,'omitnan'), mean(nValidLinear));


%% Functions
function listAER = vec_xyz2aer(listXYZ, prm)
    AzRaw = atan2(listXYZ(2,:), listXYZ(1,:));
    Az = azUnwrap(AzRaw, prm);
    horizRange = sqrt(listXYZ(1,:).^2 + listXYZ(2,:).^2);
    El = atan2(listXYZ(3,:), horizRange);
    range = sqrt(listXYZ(3,:).^2 + horizRange.^2);
    listAER = [Az; El; range];
end


function newAERinterp = AERnurecon(listAER, prm)
    upsample = 2;
    nElDec = prm.nEl * upsample;
    nAzDec = prm.nAz * upsample;
    elTicks = linspace(prm.maxEl, prm.minEl, nElDec);
    azTicks = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;

    % Filter invalid points
    curR = listAER(3,:);
    valid = curR > 0 & isfinite(curR) & curR < 200;
    az = listAER(1, valid);
    el = listAER(2, valid);
    curR = curR(valid);

    % Bilinear splat onto fine grid
    aPos = (az - azTicks(1)) / (azTicks(2) - azTicks(1)) + 1;
    ePos = (el - elTicks(1)) / (elTicks(2) - elTicks(1)) + 1;
    a0 = floor(aPos); a1 = a0 + 1;
    e0 = floor(ePos); e1 = e0 + 1;
    wa1 = aPos - a0;  wa0 = 1 - wa1;
    we1 = ePos - e0;  we0 = 1 - we1;
    a0 = max(min(a0, nAzDec), 1); a1 = max(min(a1, nAzDec), 1);
    e0 = max(min(e0, nElDec), 1); e1 = max(min(e1, nElDec), 1);

    idx = [sub2ind([nElDec nAzDec], e0, a0);
           sub2ind([nElDec nAzDec], e0, a1);
           sub2ind([nElDec nAzDec], e1, a0);
           sub2ind([nElDec nAzDec], e1, a1)];
    w = [we0.*wa0; we0.*wa1; we1.*wa0; we1.*wa1];
    r4 = repmat(curR(:)', 4, 1);
    wr4 = w .* r4;

    Rinterp = reshape(accumarray(idx(:), wr4(:), [nElDec*nAzDec 1]), nElDec, nAzDec);
    Winterp = reshape(accumarray(idx(:), w(:),   [nElDec*nAzDec 1]), nElDec, nAzDec);
    hasData = Winterp > 0;
    Rinterp(hasData) = Rinterp(hasData) ./ Winterp(hasData);

    % Gaussian spectral smoothing to fill holes
    M = double(hasData);
    [uaz, uel] = meshgrid(linspace(-0.5, 0.5, nAzDec), linspace(-0.5, 0.5, nElDec));
    sigma = 0.25;
    Gwin = ifftshift(exp(-(uaz.^2 + uel.^2) / (2*sigma^2)));
    LP_RM = real(ifft2(fft2(Rinterp .* M) .* Gwin));
    LP_M = real(ifft2(fft2(M)            .* Gwin));
    Rsmooth = zeros(nElDec, nAzDec);
    enough = LP_M > 0.01;
    Rsmooth(enough) = LP_RM(enough) ./ LP_M(enough);

    Rinterp(~hasData &  enough) = Rsmooth(~hasData & enough);
    Rinterp(~hasData & ~enough) = 0;
    Rinterp(Rinterp < 0) = 0;

    % Downsample back to original resolution
    rowIdx = round(linspace(1, nElDec, prm.nEl));
    colIdx = round(linspace(1, nAzDec, prm.nAz));
    Rout = Rinterp(rowIdx, colIdx);
    Rout(Rout <= 0 | Rout > 200) = Inf;

    [AzGrid, ElGrid] = meshgrid(azTicks(colIdx), elTicks(rowIdx));
    nCols = prm.nEl * prm.nAz;
    newAERinterp = [reshape(AzGrid, 1, nCols);
                    reshape(ElGrid, 1, nCols);
                    reshape(Rout,   1, nCols)];
end


function prm = generateMask(prm) 
% Using reference image, identify pixels associated with vehicle carrying
% LIDAR unit.  Create a mask with values of false where ranges correspond to
% the vehicle and true where ranges correspond to the scene.
    maskRangeIm = false(prm.nEl,prm.nAz);
    maskRangeIm(127:128,:)=true;       %Noise on bottom border
    maskRangeIm(118:128,178:207)=true; %Corner
    maskRangeIm(120:128,818:842)=true; %Corner
    maskRangeIm(115:128,1203:1233)=true; %Corner
    maskRangeIm(115:128,1838:1865)=true; %Corner
    maskRangeIm(120:128,2036:2048)=true; %Antenna?
    maskRangeIm(120:128,1:7)=true;       %Antenna?
    %
    totPix = numel(maskRangeIm);
    domainMask1D = reshape(maskRangeIm,[1 totPix]); %Reformat from image to row vector
    prm.domainMaskColIndxAER = find(domainMask1D);
end


function correctedAz = azUnwrap(Az,prm)
% Identify circular wrap-around error by comparing angles to a reference
% angle, based on the input PRM structure.  For instance, for a VLP-16, 
% the PRM file specifies the default azimuth decreases from a start value 
% of pi/2.
%
% Assumption:  (i) It is assumed that each LIDAR point cloud begins at PRM.azStart, 
%   in the body frame, and spins around approximately one full
%   rotation. (ii) It is assumed the error will be either one full rotation
%   too high or too low; additional multiples of a rotation are not
%   considered, nor are half rotation "slips."
%
    correctedAz = Az;
    numRows = size(Az,1);
    numCols = size(Az,2);
    nomAngle = linspace(prm.azStart,prm.azStart+prm.azDir*2*pi,numCols+1);
    nomAngle(end) = []; % Assume last point in record does not quite loop back to start
    nomAngleMat = repmat(nomAngle,numRows,1);
    angleDiff = Az-nomAngleMat;
    tooHighInd = find(angleDiff > pi/2);
    tooLowInd = find(angleDiff < -pi/2);
    correctedAz(tooHighInd) = correctedAz(tooHighInd)-2*pi;
    correctedAz(tooLowInd) = correctedAz(tooLowInd)+2*pi;
end

function vecXYZ = pc2xyz(pc)
% Convert a point-cloud structure into an XYZ array
    totPix = size(pc,1)*size(pc,2);
    vecXYZ = [reshape(pc(:,:,1),[1 totPix]); 
                 reshape(pc(:,:,2),[1 totPix]);
                 reshape(pc(:,:,3),[1 totPix]);];
end


function interpAER = vec_aerInterpSimp(listAER,prm)
% Take a vector list in AER format and resample to a specific number of
% Az/El points. This operation interpolates (e.g. reconstructs) a view
% for one location using data from a different location.
% Note: A range of NaN means non-return
    % Shift Azimuth angles to range of 0 to 2*pi
    listAER(1,:) = mod(listAER(1,:),2*pi);
    % Initialize range map interpolation
    elTicks = linspace(prm.maxEl,prm.minEl,prm.nEl);              %El Grid: radians    
    azTicks = prm.azStart+prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;   %Az Grid: radians
    Rinterp = Inf(prm.nEl,prm.nAz);                               %Initialize: Inf is default for interpolation (unless otherwise interpolated)
    % Convert AER to index
    interpAER = nan(size(listAER));
    % Interpolate
    for n=1:size(listAER,2)
        aInd = floor((listAER(1,n)-azTicks(1))/(azTicks(2)-azTicks(1)))+1;
        eInd = floor((listAER(2,n)-elTicks(1))/(elTicks(2)-elTicks(1)))+1;
        curR = listAER(3,n);
        inElRange = eInd>0 & eInd<=prm.nEl; % Make sure interpolated elevation in range of reference image
        if ~isnan(curR) && inElRange
            linearIndx = (aInd-1)*prm.nEl+eInd;
            prevR = interpAER(3,linearIndx);
            if isnan(prevR); prevR = Inf; end
            if curR < prevR % Do not replace if piexl alreayd has a nearer point
                interpAER(1:3,linearIndx)=[azTicks(aInd) elTicks(eInd) curR]';
            end
        end
    end
end 


function listAER = vec_aerInterp(listAER,prm)
% Take a vector list in AER format and resample to a specific number of
% Az/El points. This operation interpolates (e.g. reconstructs) a view
% for one location using data from a different location.
% Note: A range of NaN means non-return
    [Amat,Emat,Rmat] = vec2mataer(listAER,[prm.nEl prm.nAz]); 
    % Initialize range map interpolation
    elTicks = linspace(prm.maxEl,prm.minEl,prm.nEl);              %El Grid: radians    
    azTicks = prm.azStart+prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;   %Az Grid: radians
    Rinterp = Inf(prm.nEl,prm.nAz);                               %Initialize: Inf is default for interpolation (unless otherwise interpolated)
    % Check face direction and perform triangular interpolation

    % CHECK Triangular face direction as first step. The point order in each face
    %  should not be permuted from the original point order ... otherwise we are viewing
    %  the back of the face. If not looking at front of face, mark associated points as inconsistent.
    [isConsistSE,isConsistNW]=boxConsist(Amat,Emat);
    [rowSE,colSE] = find(isConsistSE);
    [rowNW,colNW] = find(isConsistNW);
    % Interpolate each visible face & store if results are nearer than
    %  other interpolated range values at the same Az/El
    for n=1:length(rowSE) % Scan through faces with an SE vertex
        rn = rowSE(n);
        cn = colSE(n);
        aList = [Amat(rn+1,cn+1) Amat(rn+1,cn) Amat(rn,cn+1)];
        eList = [Emat(rn+1,cn+1) Emat(rn+1,cn) Emat(rn,cn+1)];
        rList = [Rmat(rn+1,cn+1) Rmat(rn+1,cn) Rmat(rn,cn+1)];
        if prm.interpMethod == 1
            [Rinterp]=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);    %Nearest neigbor
        elseif prm.interpMethod == 2
            [Rinterp]=viewSynthesisLin(aList,eList,rList,azTicks,elTicks,Rinterp);   %Linear interp
        end
    end
    for n=1:length(rowNW) % Scan through faces with an SE vertex
        rn = rowNW(n);
        cn = colNW(n);
        aList = [Amat(rn,cn) Amat(rn+1,cn) Amat(rn,cn+1)];
        eList = [Emat(rn,cn) Emat(rn+1,cn) Emat(rn,cn+1)];
        rList = [Rmat(rn,cn) Rmat(rn+1,cn) Rmat(rn,cn+1)];
        if prm.interpMethod == 1
            [Rinterp]=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);    %Nearest neighbor
        elseif prm.interpMethod == 2
            [Rinterp]=viewSynthesisLin(aList,eList,rList,azTicks,elTicks,Rinterp);   %Linear interp
        end
    end
    %
    % Add original points to the interpolated R image, if they are nearer
    %  than other points in interpolation.  (This step is important because
    %  some valid points may not be part of a face, and those would be lost otherwise.)
    % This step is also important for the pure interpolation of the
    %   reference image
    delAz = azTicks(1)-azTicks(2);  % Az values decrease with column
    delEl = elTicks(1)-elTicks(2);  % El values decrease with row
    maxAz = azTicks(1);
    maxEl = elTicks(1);
    AzVecRnd = round((maxAz-listAER(1,:))/delAz+1);
    ElVecRnd = round((maxEl-listAER(2,:)+1e-7)/delEl+1);
    rVec = listAER(3,:);
    inRangeAzEl = find(AzVecRnd>0 & AzVecRnd<=prm.nAz & ElVecRnd>0 & ElVecRnd<=prm.nEl & rVec>0 & rVec<200);
    indxRnd = sub2ind([prm.nEl prm.nAz],ElVecRnd(inRangeAzEl),AzVecRnd(inRangeAzEl));
    replaceFlag = Rinterp(indxRnd)>rVec(inRangeAzEl);
    Rinterp(indxRnd(replaceFlag)) = rVec(inRangeAzEl(replaceFlag));
    %
    % Package results for output in list form
    [Anew,Enew]=meshgrid(azTicks,elTicks);
    nCols = prm.nAz*prm.nEl;
    listAER = [reshape(Anew,1,nCols); reshape(Enew,1,nCols); reshape(Rinterp,1,nCols);];
end %


function curView=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,curView)
% Given a face described by a row of 3 azimuth values 
%  (ALIST), a row of 3 elevations values (ELIST), and a row of 3 range 
%  values (RLIST), interpolate that face to a regular grid defined 
%  by AZTICKS and ELTICKS.  Add ranges for the face to the current view if 
%  points are "in front" (closer) compared to existing ranges in current view.
%
% NOTE - Function replaces viewSynthesisAdd (planar interpolation).
%  Nearest neighbor interpolation performed instead.
%
% Subgrid of interest
    minA=min(aList)-1e-8;
    maxA=max(aList)+1e-8;
    minE=min(eList)-1e-8;
    maxE=max(eList)+1e-8;
    azTickInd = find(azTicks>=minA & azTicks<=maxA);
    elTickInd = find(elTicks>=minE & elTicks<=maxE);
    [locA,locE]=meshgrid(azTicks(azTickInd),elTicks(elTickInd));
    % Compute edge vectors
    v12 = [aList(2)-aList(1) eList(2)-eList(1)];
    v23 = [aList(3)-aList(2) eList(3)-eList(2)];
    v31 = [aList(1)-aList(3) eList(1)-eList(3)];
    v12mag = norm(v12);
    v23mag = norm(v23);
    v31mag = norm(v31);
    u12 = v12/v12mag;
    u23 = v23/v23mag;
    u31 = v31/v31mag;
    vert1 = u12(1)*(locA-aList(1))+u12(2)*(locE-eList(1))<v12mag/2 & -u31(1)*(locA-aList(1))-u31(2)*(locE-eList(1))<=v31mag/2; %Closer to vertex 1 than other vertices
    vert2 = -u12(1)*(locA-aList(2))-u12(2)*(locE-eList(2))<=v12mag/2 & u23(1)*(locA-aList(2))+u23(2)*(locE-eList(2))<v23mag/2; %Closer to vertex 2 than other vertices
    vert3 = -u23(1)*(locA-aList(3))-u23(2)*(locE-eList(3))<=v23mag/2 & u31(1)*(locA-aList(3))+u31(2)*(locE-eList(3))<v31mag/2; %Closer to vertex 3 than other vertices
    % Set range to nearest neighbor
    inMask=inTriangle(aList,eList,locA,locE); %inMask describes being inside triangle, vert_j describes nearest vertex -> both should be meshgrid arrays
    locR = Inf(size(locA));
    locI = Inf(size(locA));
    locR(inMask&vert1)=rList(1);
    locR(inMask&vert2)=rList(2);
    locR(inMask&vert3)=rList(3);
    % Update curview ranges with points inside face
    if length(elTickInd)*length(azTickInd)>0 % only update if points # of points inside face is above zero
        blockR = curView(elTickInd,azTickInd);
        newNearerMask = find(blockR>locR);
        blockR(newNearerMask)=locR(newNearerMask);
        curView(elTickInd,azTickInd)=blockR;
    end
end

function curView=viewSynthesisLin(aList,eList,rList,azTicks,elTicks,curView)
% NOTE - Replaced by viewSynthesisNN (which does nearest-neighbor lookup rather than a planar interpolation)
% Given a face described by a row of 3 azimuth values 
% (ALIST), a row of 3 elevations values (ELIST), and a row of 3 range 
% values (RLIST), interpolate that face to a regular grid defined 
% by AZTICKS and ELTICKS.  Add ranges for the face to the current view if 
% points are "in front" (closer) compared to existing ranges in current view.

    % Equation for plane through face
    A = [aList' eList' ones(3,1)];
    b = rList';
    planeCoef = A\b;
    % Subgrid of interest
    minA=min(aList);
    maxA=max(aList);
    minE=min(eList);
    maxE=max(eList);
    azTickInd = find(azTicks>=minA & azTicks<=maxA);
    elTickInd = find(elTicks>=minE & elTicks<=maxE);
    [locA,locE]=meshgrid(azTicks(azTickInd),elTicks(elTickInd));
    % Identify points inside face
    inMask=inTriangle(aList,eList,locA,locE);
    inMaskIndx=find(inMask);
    locR = Inf(size(locA));
    locR(inMaskIndx)= locA(inMaskIndx)*planeCoef(1)+locE(inMaskIndx)*planeCoef(2)+planeCoef(3); % use interpolation to compute R on face
    % Update curview ranges with points inside face
    if length(elTickInd)*length(azTickInd)>0 % only update if points # of points inside face is above zero
        curView(elTickInd,azTickInd)=min(curView(elTickInd,azTickInd),locR);
    end
end


function [isConsistSE,isConsistNW]=boxConsist(aGrid,eGrid)
% Check for faces that are no longer facing the sensor after a rigid transformation.
%  Assume that the grid originally possessed regular Az/El spacing but underwent
%  a rigid transformation (to a new sensor pose) such that Az/El grid
%  is no longer regular.  If points are inconsistent, sensor is now looking
%  at back of a face of 3 points rather than the front.
%
%  For every set of 4 points that originally defined a "square" in Az/El
%  space, consider two faces: an upper-left (Northwest or NW) set of three
%  points and a lower-right (Southeast or SE) set of three points.
% 
%  The transformed Az/El coordinates are given by AGRID and EGRID, where
%  the order of the points matches the pre-transfor grid.  The outputs
%  ISCONSISTENT have dimension one less (in each direction) than the GRIDS
%  and represent whether each corresponding vertex is associated with a
%  valid triangle.

    % Define edge differences.  Note that the Az and El values by default are
    %  decreasing as the index increases.  So create differences that should by
    %  default be positive.
    dAlat = aGrid(:,1:end-1)-aGrid(:,2:end);  
    dEvrt = eGrid(1:end-1,:)-eGrid(2:end,:);  
    % Because of transform, Az changes not only in lateral direction, but in
    % vertical; El now changes not just in vertical but also lateral.  Copy
    % difference directions for lateral/vertical relationships from above.  In
    % other words:  lateral differences are to (j-1) from (j); vertical diff
    % are from from (i-1) to (i).
    dElat = eGrid(:,1:end-1)-eGrid(:,2:end);   
    dAvrt = aGrid(1:end-1,:)-aGrid(2:end,:);  
    %
    % Set up SE cross product in form a*azHat+b*ElHat is new lateral vector
    % and c*azHat+d*elHat is vertical vector
    a = dAlat(2:end,:); % eliminate top row, because no SE vertices can exist in top row
    b = dElat(2:end,:);
    c = dAvrt(:,2:end); % eliminate first col, because no SE vertices can exist in first col
    d = dEvrt(:,2:end);
    crossSE = a.*d-b.*c; % On original grid: cross vertical is outward (positive)
    isConsistSE = crossSE>0;
    %
    % Set up NW cross product in form a*azHat+b*ElHat is new lateral vector
    % and c*azHat+d*elHat is vertical vector
    a = dAlat(1:end-1,:); % eliminate last row, because no NW vertices can exist in bottom row
    b = dElat(1:end-1,:);
    c = dAvrt(:,1:end-1); % eliminate last col, because no NW vertices can exist in last col
    d = dEvrt(:,1:end-1);
    crossNW = a.*d-b.*c; % On original grid: cross vertical is outward (positive)
    isConsistNW = crossNW>0;
end

function inMask=inTriangle(vx,vy,xGrid,yGrid)
% Given a set of 3 x&y coordinates (row vectors) for vertices and a mesghrid of xy
% values, test each value on the grid to see if it is in the triangle.
% (Use a closed set, where border is included as "inside")
    vMat = [vx; vy; zeros(1,3)];
    khat = [0 0 1]; % vector "out of page"
    v12 = vMat(:,2)-vMat(:,1); %vectors between vertices (edges)
    v23 = vMat(:,3)-vMat(:,2);
    v31 = vMat(:,1)-vMat(:,3);
    nv12 = cross(v12,khat); %vectors normal to edges
    nv23 = cross(v23,khat);
    nv31 = cross(v31,khat);
    inside12 = (nv12(1)*(xGrid-vx(1))+nv12(2)*(yGrid-vy(1)))*sign(dot(nv12,-v31))>=-1e-5; % If dot product of grid point (relative to vertex 1) with v1v2 normal has same sign as dot product of vertex 3 relative to v1, then points *can be* inside triangle
    inside23 = (nv23(1)*(xGrid-vx(2))+nv23(2)*(yGrid-vy(2)))*sign(dot(nv23,-v12))>=-1e-5; % Note: add a bit of margin (-1e-5 rather than zero) to deal with rounding errors
    inside31 = (nv31(1)*(xGrid-vx(3))+nv31(2)*(yGrid-vy(3)))*sign(dot(nv31,-v23))>=-1e-5; 
    inMask = inside12&inside23&inside31;
end
