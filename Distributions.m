prm.refFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';
prm.refFrameNum = 1;
prm.newFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';%
prm.newFrameNum = 45; % Total 60
%  Set rotation matrix and translation vector from current point cloud to reference
%  from ICP: 
% pcNew=pointCloud(newXYZraw);  % NEEDS COMPUTER VISION TOOLBOX
% pcRef=pointCloud(refXYZ); 
% tform = pcregistericp(pcNew,pcRef,Metric="planeToPlane");
% prm.R = tform.Rotation;
% prm.delta = tform.Translation';
% prm.R = [1.0000    0.0014    0.0005; -0.0014    1.0000   -0.0011; -0.0005    0.0011    1.0000]; %Frame 1->11
% prm.delta = [-8.4330    0.0570    0.0702]';

%  lidar model description
prm.azStart = 2*pi;              % Nominal start angle for each frame (rad)
prm.azDir = -1;                  % Sign of azimuth change (positive for increasing angle; negative for decreasing angle)
prm.minEl =-11.27*pi/180;        % Min elevation of scan (rad)
prm.maxEl = 10.63*pi/180;        % Max elevation of scan (rad)
% interpolation 
prm.interpMethod = 1;            % Options: 0 is no interpolation, 1 is nearest neighobr, 2 is planar fit

%%% Load two images
%  Input format is cell array called "pc" with XYZ point cloud as 3D array in each cell
load(prm.refFileName);  
refPC = pc{prm.refFrameNum};
if strcmp(prm.refFileName,prm.newFileName)
    % do nothing -- same file
else
    % load second file
    load(prm.newFileName);
end
newPC = pc{prm.newFrameNum};
newXYZraw = pc2xyz(newPC);                                          % Convert point cloud to XYZ array (1D for each coordinate)
refXYZ = pc2xyz(refPC);                                             % Convert point cloud to XYZ array

pcNew = pointCloud(newXYZraw');
pcRef = pointCloud(refXYZ');
tform = pcregistericp(pcNew, pcRef, Metric="planeToPlane");
prm.R = tform.Rotation;
prm.delta = tform.Translation';
disp(prm.R);
disp(prm.delta);

%  Reference size
prm.nEl = size(refPC,1);         % Both images assumed to have same dimension (note -- this is not true for older VLP16 data!)
prm.nAz = size(refPC,2); 
%  Identify ego vehicle
prm = generateMask(prm);         % Apply mask to parts of scene that are associated with the ego vehicle

%%% Registration: Resample clouds to common (Az,El) grid
%  Current image
newXYZreg = prm.R*double(newXYZraw(1:3,:))+...
    repmat(prm.delta,[1 size(newXYZraw,2)]);                        % Apply pose transform to register current image to reference
newAERreg = vec_xyz2aer(newXYZreg,prm);
newAERreg(3,prm.domainMaskColIndxAER) = -1;                         % Exclude raypaths describing ego vehicle, setting range to -1 >> negative ranges should be impossible
newAERinterpNN = vec_aerInterpMeth(newAERreg,prm);
prm.interpMethod = 2;
newAERinterpPlanar = vec_aerInterpMeth(newAERreg,prm);
newAERinterpSplat = AERnurecon(newAERreg, prm);
% Spatial interpolation methods (scatteredInterpolant)
newAERinterpSpatialNearest = AERscatteredInterp(newAERreg, prm, 'nearest');
newAERinterpSpatialLinear  = AERscatteredInterp(newAERreg, prm, 'linear');
newAERinterpSpatialNatural = AERscatteredInterp(newAERreg, prm, 'natural');
% Reference image
refAER = vec_xyz2aer(refXYZ,prm);
refAER(3,prm.domainMaskColIndxAER) = -1;                            % Exclude raypaths describing ego vehicle, setting range to -1 >> negative ranges should be impossible
refAERinterp = vec_aerInterpMeth(refAER,prm);      



figure(); clf;
newRNN = newAERinterpNN(3,:);
newRNN(find(newRNN<0 | isinf(newRNN))) = NaN;
newRNN = reshape(newRNN,[prm.nEl,prm.nAz]);

newRPlanar = newAERinterpPlanar(3,:);
newRPlanar(find(newRPlanar<0 | isinf(newRPlanar))) = NaN;
newRPlanar = reshape(newRPlanar,[prm.nEl,prm.nAz]);

newRSplat = newAERinterpSplat(3,:);
newRSplat(find(newRSplat<0 | isinf(newRSplat))) = NaN;
newRSplat = reshape(newRSplat,[prm.nEl,prm.nAz]);

newRSpatialNearest = newAERinterpSpatialNearest(3,:);
newRSpatialNearest(find(newRSpatialNearest<0 | isinf(newRSpatialNearest))) = NaN;
newRSpatialNearest = reshape(newRSpatialNearest,[prm.nEl,prm.nAz]);

newRSpatialLinear = newAERinterpSpatialLinear(3,:);
newRSpatialLinear(find(newRSpatialLinear<0 | isinf(newRSpatialLinear))) = NaN;
newRSpatialLinear = reshape(newRSpatialLinear,[prm.nEl,prm.nAz]);

newRSpatialNatural = newAERinterpSpatialNatural(3,:);
newRSpatialNatural(find(newRSpatialNatural<0 | isinf(newRSpatialNatural))) = NaN;
newRSpatialNatural = reshape(newRSpatialNatural,[prm.nEl,prm.nAz]);



refR = refAERinterp(3,:);
refR(find(refR<0 | isinf(refR))) = NaN;
refR = reshape(refR,[prm.nEl,prm.nAz]);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRNN) & ~isnan(newRNN);
dRNN = newRNN(valid)-refR(valid);
stdevNN = std(dRNN);
meanNN = mean(dRNN);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRPlanar) & ~isnan(newRPlanar);
dRPlanar = newRPlanar(valid)-refR(valid);
stdevPlanar = std(dRPlanar);
meanPlanar = mean(dRPlanar);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRSplat) & ~isnan(newRSplat);
dRSplat = newRSplat(valid)-refR(valid);
stdevSplat = std(dRSplat);
meanSplat = mean(dRSplat);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRSpatialNearest) & ~isnan(newRSpatialNearest);
dRSpatialNearest = newRSpatialNearest(valid)-refR(valid);
stdevSpatialNearest = std(dRSpatialNearest);
meanSpatialNearest = mean(dRSpatialNearest);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRSpatialLinear) & ~isnan(newRSpatialLinear);
dRSpatialLinear = newRSpatialLinear(valid)-refR(valid);
stdevSpatialLinear = std(dRSpatialLinear);
meanSpatialLinear = mean(dRSpatialLinear);

valid = isfinite(refR) & ~isnan(refR) & isfinite(newRSpatialNatural) & ~isnan(newRSpatialNatural);
dRSpatialNatural = newRSpatialNatural(valid)-refR(valid);
stdevSpatialNatural = std(dRSpatialNatural);
meanSpatialNatural = mean(dRSpatialNatural);

% Splat
subplot(2,2,1);
histogram(dRSplat, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)');
ylabel('Probability Density');
title('Splat');
grid on;

% Nearest Neighbor
subplot(2,2,2);
histogram(dRSpatialNearest, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)');
ylabel('Probability Density');
title('Nearest Neighbor');
grid on;

% Linear
subplot(2,2,3);
histogram(dRSpatialLinear, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)');
ylabel('Probability Density');
title('Linear');
grid on;

% Natural Neighbor
subplot(2,2,4);
histogram(dRSpatialNatural, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)');
ylabel('Probability Density');
title('Natural Neighbor');
grid on;
sgtitle('Range Error Distribution (45 Frames - 35.5 m)');







function newAERinterp = AERscatteredInterp(listAER, prm, method)

    r = listAER(3,:);
    valid = r > 0 & isfinite(r) & r < 200;

    az = listAER(1,valid);
    el = listAER(2,valid);
    r  = r(valid);

    % Wrap azimuth
    az = mod(az,2*pi);

    % Target grid
    elTicks = linspace(prm.maxEl, prm.minEl, prm.nEl);
    azTicks = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    [AzGrid,ElGrid] = meshgrid(azTicks,elTicks);

    % Interpolant
    F = scatteredInterpolant(az',el',r',method,'none');

    Rout = F(AzGrid,ElGrid);
    Rout(~isfinite(Rout)) = Inf;

    nCols = prm.nEl * prm.nAz;

    newAERinterp = [
        reshape(AzGrid,1,nCols);
        reshape(ElGrid,1,nCols);
        reshape(Rout,1,nCols)
    ];
end

function vecXYZ = pc2xyz(pc)
% Convert a point-cloud structure into an XYZ array
    totPix = size(pc,1)*size(pc,2);
    vecXYZ = [reshape(pc(:,:,1),[1 totPix]); 
                 reshape(pc(:,:,2),[1 totPix]);
                 reshape(pc(:,:,3),[1 totPix]);];
end

function listXYZ = vec_aer2xyz(listAER)
    az = listAER(1,:);
    el = listAER(2,:);
    r = listAER(3,:);
    x = r .* cos(el) .* cos(az);
    y = r .* cos(el) .* sin(az);
    z = r .* sin(el);
    listXYZ = [x;y;z];
end

function listAER = vec_xyz2aer(listXYZ,prm)
% Take a vector list (3 rows are XYZ coordinates, N columns are different
% points) and convert each vector to Az/El coord.  
    AzRaw = atan2(listXYZ(2,:),listXYZ(1,:));
    Az = azUnwrap(AzRaw,prm);
    horizRange = sqrt(listXYZ(1,:).^2+listXYZ(2,:).^2);
    El = atan2(listXYZ(3,:),horizRange);
    range = sqrt(listXYZ(3,:).^2+horizRange.^2);
    listAER = [Az; El; range];
end

function [matA,matE,matR] = vec2mataer(vecAER,imSize)
% Take a vector list (3 rows are AER coordinates, N columns are different
% points) and convert to structure with three matrix fields of size IMSIZE:
% MATAER.A, MATAER.E, and MATAER.R.  
%
% Overloaded to work with both AER and AERI structures
    matA = reshape(vecAER(1,:),imSize(1:2));
    matE = reshape(vecAER(2,:),imSize(1:2));
    matR = reshape(vecAER(3,:),imSize(1:2));
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


% INTERPOLATION METHODS
function interpAER = vec_aerInterpMeth(listAER,prm)
    switch prm.interpMethod
        case 0
            interpAER = vec_aerInterpSimp(listAER,prm);
        case 1
            interpAER = vec_aerInterp(listAER,prm);
        case 2
            interpAER = vec_aerInterp(listAER,prm); % further details of interpolation inside this function
        case 3
            interpAER = vec_aerInterpZeroPad(listAER,prm);
        case 4
            interpAER = vec_aerInterpZeroHold(listAER,prm);
    end
end

function newAERinterp = AERnurecon(listAER, prm)
    upsample = 3;
    nElDec = prm.nEl * upsample;
    nAzDec = prm.nAz * upsample;
    elTicks = linspace(prm.maxEl, prm.minEl, nElDec);
    azTicks = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;

    % Filter invalid points
    curR = listAER(3,:);
    valid = curR > 0 & isfinite(curR) & curR < 200;
    az = listAER(1,valid);
    el = listAER(2,valid);
    curR = curR(valid);

    % Bilinear splat onto fine grid using accumarray
    aPos = (az - azTicks(1)) / (azTicks(2)-azTicks(1)) + 1;
    ePos = (el - elTicks(1)) / (elTicks(2)-elTicks(1)) + 1;
    a0 = floor(aPos); a1 = a0+1;
    e0 = floor(ePos); e1 = e0+1;
    wa1 = aPos-a0; wa0 = 1-wa1;
    we1 = ePos-e0; we0 = 1-we1;
    a0 = max(min(a0,nAzDec),1); a1 = max(min(a1,nAzDec),1);
    e0 = max(min(e0,nElDec),1); e1 = max(min(e1,nElDec),1);
    idx = [sub2ind([nElDec nAzDec],e0,a0);
           sub2ind([nElDec nAzDec],e0,a1);
           sub2ind([nElDec nAzDec],e1,a0);
           sub2ind([nElDec nAzDec],e1,a1)];
    w  = [we0.*wa0; we0.*wa1; we1.*wa0; we1.*wa1];
    r4 = repmat(curR(:)', 4, 1);
    wr4 = w .* r4;
    Rinterp = reshape(accumarray(idx(:), wr4(:), [nElDec*nAzDec 1]), nElDec, nAzDec);
    Winterp = reshape(accumarray(idx(:), w(:),   [nElDec*nAzDec 1]), nElDec, nAzDec);
    hasData = Winterp > 0;
    Rinterp(hasData) = Rinterp(hasData) ./ Winterp(hasData);

    % Normalized Gaussian spectral smoothing — no ringing, no background bias
    M = double(hasData);
    [uaz, uel] = meshgrid(linspace(-0.5,0.5,nAzDec), linspace(-0.5,0.5,nElDec));
    sigma = 0.25;
    Gwin  = ifftshift(exp(-(uaz.^2 + uel.^2) / (2*sigma^2)));
    LP_RM = real(ifft2(fft2(Rinterp .* M) .* Gwin));
    LP_M  = real(ifft2(fft2(M)            .* Gwin));
    Rsmooth = zeros(nElDec, nAzDec);
    enough  = LP_M > 0.01;
    Rsmooth(enough) = LP_RM(enough) ./ LP_M(enough);

    % Final fixes - holes
    Rinterp(~hasData & enough)  = Rsmooth(~hasData & enough);
    Rinterp(~hasData & ~enough) = 0;
    Rinterp(Rinterp < 0) = 0;

    % Format output
    rowIdx = round(linspace(1,nElDec,prm.nEl));
    colIdx = round(linspace(1,nAzDec,prm.nAz));
    Rout = Rinterp(rowIdx, colIdx);
    Rout(Rout <= 0 | Rout > 200) = Inf;
    [AzGrid, ElGrid] = meshgrid(azTicks(colIdx), elTicks(rowIdx));
    nCols = prm.nEl * prm.nAz;
    newAERinterp = [reshape(AzGrid,1,nCols); reshape(ElGrid,1,nCols); reshape(Rout,1,nCols)];
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
        if ~isnan(curR) & inElRange
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

function listAER = vec_aerInterpZeroPad(listAER,prm)
    % Initialize range map (finer)
    nElDec = prm.nEl * 10;
    nAzDec = prm.nAz * 10;
    elTicks = linspace(prm.maxEl,prm.minEl, nElDec);              %El Grid: radians    
    azTicks = prm.azStart+prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;     %Az Grid: radians
    RinterpDec = Inf(nElDec,nAzDec);
    % Fit the original points to the finer grid
    aInd = round((listAER(1,:)-azTicks(1))/(azTicks(2)-azTicks(1))) + 1;
    eInd = round((listAER(2,:)-elTicks(1))/(elTicks(2)-elTicks(1))) + 1;
    curR = listAER(3,:);
    % Check within bounds
    aInd = max(min(aInd, nAzDec), 1);
    eInd = max(min(eInd, nElDec), 1);
    % Assign points to finer grid
    for n = 1:size(listAER,2)
        if(RinterpDec(eInd(n),aInd(n)) == 0)
            RinterpDec(eInd(n),aInd(n)) = curR(n);
        else
            RinterpDec(eInd(n), aInd(n)) = min(RinterpDec(eInd(n), aInd(n)), curR(n));
        end    
    end
    % Create filter to lowpass
    mask = isfinite(RinterpDec);
    % Zeropad
    RinterpDec(~mask) = 0;
    lowpass = ones(20,20) / 400;
    % LPF
    RinterpDec = conv2(RinterpDec,lowpass,'same') ./ conv2(mask, lowpass, 'same');
    % Downsample
    Rinterp = RinterpDec(1:10:end, 1:10:end);
    [AzGrid, ElGrid] = meshgrid(azTicks(1:10:end), elTicks(1:10:end));
    listAER = [reshape(AzGrid,1,[]); reshape(ElGrid,1,[]); reshape(Rinterp,1,[])];
end


function listAER = vec_aerInterpZeroHold(listAER,prm)
    % Initialize range map (finer)
    nElDec = prm.nEl * 10;
    nAzDec = prm.nAz * 10;
    elTicks = linspace(prm.maxEl,prm.minEl, nElDec);              %El Grid: radians    
    azTicks = prm.azStart+prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;     %Az Grid: radians
    RinterpDec = Inf(nElDec,nAzDec);
    % Fit the original points to the finer grid
    aInd = round((listAER(1,:)-azTicks(1))/(azTicks(2)-azTicks(1))) + 1;
    eInd = round((listAER(2,:)-elTicks(1))/(elTicks(2)-elTicks(1))) + 1;
    curR = listAER(3,:);
    % Check within bounds
    aInd = max(min(aInd, nAzDec), 1);
    eInd = max(min(eInd, nElDec), 1);
    % Create circle zero hold
    r = 0.5 * 10;
    [x, y] = meshgrid(-r:r, -r:r);
    circle = (x.^2 + y.^2 < r^2);

    for n = 1:size(listAER,2)
        el = eInd(n) + y(:);
        az = aInd(n) + x(:);
        circleVector = circle(:);
        % Within Bounds of Indices
        valid = (circleVector & el >= 1 & el <= nElDec & az >= 1 & az <= nAzDec);
        el = el(valid);
        az = az(valid);

        for i = 1:length(el)
            RinterpDec(el(i), az(i)) = min(RinterpDec(el(i), az(i)), curR(n));
        end
    end
    % Create filter to lowpass
    mask = isfinite(RinterpDec);
    % Zeropad
    RinterpDec(~mask) = 0;
    % LPF
    lowpass = ones(10,10) / 100;
    RinterpDec = conv2(RinterpDec,lowpass,'same') ./ conv2(mask, lowpass, 'same');
    % Downsample
    Rinterp = RinterpDec(1:10:end, 1:10:end);
    [AzGrid, ElGrid] = meshgrid(azTicks(1:10:end), elTicks(1:10:end));
    listAER = [reshape(AzGrid,1,[]); reshape(ElGrid,1,[]); reshape(Rinterp,1,[])];
end


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


function AERmask = maskAER(refAER)
    AERmask = Inf(size(refAER));
    AERmask(isfinite(refAER)) = 1;
end

function listAERfft = AERfft(refAER, prm)
    refAER(isnan(refAER) | isinf(refAER)) = 256;
    listAERfft = fft2(refAER);
end

function listAERrecon = AERifft(refAERfft, prm)
    listAERrecon = ifft2(refAERfft);
end



%%% VIZUALIZATION
function vizAzEl(imAER,prm)
    [Amat,Emat,Rmat] = vec2mataer(imAER,[prm.nEl size(imAER,2)/prm.nEl]); 
    image(3./Rmat*255*2);
end
