% Distributions.m
% Computes range error distributions for a single frame pair
% Runs 6 methods: Triangle NN, Triangle Planar, Splat, Spatial Nearest, Spatial Linear, Spatial Natural
% Plots error histograms (PDF) for Splat, Nearest, Linear, Natural
clear; clc;

% Frame config
prm.refFileName  = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';
prm.refFrameNum  = 1;
prm.newFileName  = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';
prm.newFrameNum  = 45; % frames 1->45 ~= 35.5m

% Sensor parameters
prm.azStart      = 2*pi;
prm.azDir        = -1;
prm.minEl        = -11.27*pi/180;
prm.maxEl        =  10.63*pi/180;
prm.interpMethod = 1;

% Load frames
load(prm.refFileName);
refPC = pc{prm.refFrameNum};
if ~strcmp(prm.refFileName, prm.newFileName)
    load(prm.newFileName);
end
newPC     = pc{prm.newFrameNum};
newXYZraw = pc2xyz(newPC);
refXYZ    = pc2xyz(refPC);

% ICP registration
pcNew     = pointCloud(newXYZraw');
pcRef     = pointCloud(refXYZ');
tform     = pcregistericp(pcNew, pcRef, Metric="planeToPlane");
prm.R     = tform.Rotation;
prm.delta = tform.Translation';
disp(prm.R); disp(prm.delta);

prm.nEl = size(refPC, 1);
prm.nAz = size(refPC, 2);
prm     = generateMask(prm);

% Transform new frame into reference coordinates
newXYZreg = prm.R * double(newXYZraw(1:3,:)) + repmat(prm.delta, [1 size(newXYZraw,2)]);
newAERreg  = vec_xyz2aer(newXYZreg, prm);
newAERreg(3, prm.domainMaskColIndxAER) = -1;

% Run methods
prm.interpMethod   = 1;
newAERinterpNN     = vec_aerInterpMeth(newAERreg, prm);
prm.interpMethod   = 2;
newAERinterpPlanar = vec_aerInterpMeth(newAERreg, prm);
newAERinterpSplat  = AERnurecon(newAERreg, prm);

newAERinterpSpatialNearest = AERscatteredInterp(newAERreg, prm, 'nearest');
newAERinterpSpatialLinear  = AERscatteredInterp(newAERreg, prm, 'linear');
newAERinterpSpatialNatural = AERscatteredInterp(newAERreg, prm, 'natural');

% Reference frame
refAER = vec_xyz2aer(refXYZ, prm);
refAER(3, prm.domainMaskColIndxAER) = -1;
prm.interpMethod = 1;
refAERinterp = vec_aerInterpMeth(refAER, prm);

% Extract and clean range images
toR = @(aer) reshape(aer(3,:), prm.nEl, prm.nAz);
clean = @(R) setnan(R, R < 0 | isinf(R));

refR               = toR(refAERinterp);    refR(refR<0|isinf(refR))=NaN;
newRNN             = toR(newAERinterpNN);  newRNN(newRNN<0|isinf(newRNN))=NaN;
newRPlanar         = toR(newAERinterpPlanar); newRPlanar(newRPlanar<0|isinf(newRPlanar))=NaN;
newRSplat          = toR(newAERinterpSplat);  newRSplat(newRSplat<0|isinf(newRSplat))=NaN;
newRSpatialNearest = toR(newAERinterpSpatialNearest); newRSpatialNearest(newRSpatialNearest<0|isinf(newRSpatialNearest))=NaN;
newRSpatialLinear  = toR(newAERinterpSpatialLinear);  newRSpatialLinear(newRSpatialLinear<0|isinf(newRSpatialLinear))=NaN;
newRSpatialNatural = toR(newAERinterpSpatialNatural); newRSpatialNatural(newRSpatialNatural<0|isinf(newRSpatialNatural))=NaN;

% Compute error per method
vmask = @(A, B) isfinite(A) & isfinite(B);

v = vmask(refR, newRNN);             dRNN             = newRNN(v)             - refR(v);
v = vmask(refR, newRPlanar);         dRPlanar         = newRPlanar(v)         - refR(v);
v = vmask(refR, newRSplat);          dRSplat          = newRSplat(v)          - refR(v);
v = vmask(refR, newRSpatialNearest); dRSpatialNearest = newRSpatialNearest(v) - refR(v);
v = vmask(refR, newRSpatialLinear);  dRSpatialLinear  = newRSpatialLinear(v)  - refR(v);
v = vmask(refR, newRSpatialNatural); dRSpatialNatural = newRSpatialNatural(v) - refR(v);

% Summary
fprintf('Method             Mean(m)   Std(m)\n');
fprintf('Triangle NN        %6.3f    %6.3f\n', mean(dRNN),             std(dRNN));
fprintf('Triangle Planar    %6.3f    %6.3f\n', mean(dRPlanar),         std(dRPlanar));
fprintf('Splat              %6.3f    %6.3f\n', mean(dRSplat),          std(dRSplat));
fprintf('Spatial Nearest    %6.3f    %6.3f\n', mean(dRSpatialNearest), std(dRSpatialNearest));
fprintf('Spatial Linear     %6.3f    %6.3f\n', mean(dRSpatialLinear),  std(dRSpatialLinear));
fprintf('Spatial Natural    %6.3f    %6.3f\n', mean(dRSpatialNatural), std(dRSpatialNatural));

% Plot histograms
figure; clf;
subplot(2,2,1);
histogram(dRSplat, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)'); ylabel('Probability Density'); title('Splat'); grid on;

subplot(2,2,2);
histogram(dRSpatialNearest, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)'); ylabel('Probability Density'); title('Nearest Neighbor'); grid on;

subplot(2,2,3);
histogram(dRSpatialLinear, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)'); ylabel('Probability Density'); title('Linear'); grid on;

subplot(2,2,4);
histogram(dRSpatialNatural, 100, 'Normalization', 'pdf');
xlabel('Range Error (m)'); ylabel('Probability Density'); title('Natural Neighbor'); grid on;

sgtitle(sprintf('Range Error Distribution (frame %d — %.1fm)', prm.newFrameNum, norm(prm.delta)));


%% Functions

% Wraps scatteredInterpolant for AER input onto a regular grid
% In: listAER [3xN], prm, method string ('nearest','linear','natural')
% Out: newAERinterp [3 x nEl*nAz]
function newAERinterp = AERscatteredInterp(listAER, prm, method)
    r     = listAER(3,:);
    valid = r > 0 & isfinite(r) & r < 200;
    az    = mod(listAER(1, valid), 2*pi);
    el    = listAER(2, valid);
    r     = r(valid);
    elTicks = linspace(prm.maxEl, prm.minEl, prm.nEl);
    azTicks = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    [AzGrid, ElGrid] = meshgrid(azTicks, elTicks);
    F    = scatteredInterpolant(az', el', r', method, 'none');
    Rout = F(AzGrid, ElGrid);
    Rout(~isfinite(Rout)) = Inf;
    nCols = prm.nEl * prm.nAz;
    newAERinterp = [reshape(AzGrid,1,nCols); reshape(ElGrid,1,nCols); reshape(Rout,1,nCols)];
end

% Splats points onto oversampled grid with bilinear weights, fills holes via Gaussian spectral smoothing
% In: listAER [3xN], prm  Out: newAERinterp [3 x nEl*nAz]
% Note: upsample=3 here vs 2 in benchmark scripts
function newAERinterp = AERnurecon(listAER, prm)
    upsample = 3;
    nElDec = prm.nEl * upsample;
    nAzDec = prm.nAz * upsample;
    elTicks = linspace(prm.maxEl, prm.minEl, nElDec);
    azTicks = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;
    curR  = listAER(3,:);
    valid = curR > 0 & isfinite(curR) & curR < 200;
    az    = listAER(1, valid);
    el    = listAER(2, valid);
    curR  = curR(valid);
    aPos=(az-azTicks(1))/(azTicks(2)-azTicks(1))+1;
    ePos=(el-elTicks(1))/(elTicks(2)-elTicks(1))+1;
    a0=floor(aPos); a1=a0+1; e0=floor(ePos); e1=e0+1;
    wa1=aPos-a0; wa0=1-wa1; we1=ePos-e0; we0=1-we1;
    a0=max(min(a0,nAzDec),1); a1=max(min(a1,nAzDec),1);
    e0=max(min(e0,nElDec),1); e1=max(min(e1,nElDec),1);
    idx=[sub2ind([nElDec nAzDec],e0,a0); sub2ind([nElDec nAzDec],e0,a1);
         sub2ind([nElDec nAzDec],e1,a0); sub2ind([nElDec nAzDec],e1,a1)];
    w=[we0.*wa0; we0.*wa1; we1.*wa0; we1.*wa1];
    wr4=w.*repmat(curR(:)',4,1);
    Rinterp=reshape(accumarray(idx(:),wr4(:),[nElDec*nAzDec 1]),nElDec,nAzDec);
    Winterp=reshape(accumarray(idx(:),w(:),  [nElDec*nAzDec 1]),nElDec,nAzDec);
    hasData=Winterp>0;
    Rinterp(hasData)=Rinterp(hasData)./Winterp(hasData);
    M=double(hasData);
    [uaz,uel]=meshgrid(linspace(-0.5,0.5,nAzDec),linspace(-0.5,0.5,nElDec));
    Gwin=ifftshift(exp(-(uaz.^2+uel.^2)/(2*0.25^2)));
    LP_RM=real(ifft2(fft2(Rinterp.*M).*Gwin));
    LP_M=real(ifft2(fft2(M).*Gwin));
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

% Dispatches to interpolation method selected by prm.interpMethod
% In: listAER [3xN], prm (0=simple, 1=triNN, 2=triLin)
% Out: interpAER [3 x nEl*nAz]
function interpAER = vec_aerInterpMeth(listAER, prm)
    switch prm.interpMethod
        case 0; interpAER = vec_aerInterpSimp(listAER, prm);
        case 1; interpAER = vec_aerInterp(listAER, prm);
        case 2; interpAER = vec_aerInterp(listAER, prm);
    end
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

% Floor-index nearest-pixel assignment with no hole filling
% In: listAER [3xN], prm  Out: interpAER [3 x nEl*nAz]
function interpAER = vec_aerInterpSimp(listAER, prm)
    listAER(1,:)=mod(listAER(1,:),2*pi);
    elTicks=linspace(prm.maxEl,prm.minEl,prm.nEl);
    azTicks=prm.azStart+prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    interpAER=nan(size(listAER));
    for n=1:size(listAER,2)
        aInd=floor((listAER(1,n)-azTicks(1))/(azTicks(2)-azTicks(1)))+1;
        eInd=floor((listAER(2,n)-elTicks(1))/(elTicks(2)-elTicks(1)))+1;
        curR=listAER(3,n);
        inElRange=eInd>0&eInd<=prm.nEl;
        if ~isnan(curR)&&inElRange
            linearIndx=(aInd-1)*prm.nEl+eInd;
            prevR=interpAER(3,linearIndx);
            if isnan(prevR); prevR=Inf; end
            if curR<prevR
                interpAER(1:3,linearIndx)=[azTicks(aInd) elTicks(eInd) curR]';
            end
        end
    end
end

% Triangle-based forward-warp interpolation with face visibility check and isolated point snapping
% In: listAER [3xN], prm (interpMethod: 1=NN, 2=planar)
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
        if prm.interpMethod==1; Rinterp=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);
        else; Rinterp=viewSynthesisLin(aList,eList,rList,azTicks,elTicks,Rinterp); end
    end
    for n=1:length(rowNW)
        rn=rowNW(n); cn=colNW(n);
        aList=[Amat(rn,cn) Amat(rn+1,cn) Amat(rn,cn+1)];
        eList=[Emat(rn,cn) Emat(rn+1,cn) Emat(rn,cn+1)];
        rList=[Rmat(rn,cn) Rmat(rn+1,cn) Rmat(rn,cn+1)];
        if prm.interpMethod==1; Rinterp=viewSynthesisNN(aList,eList,rList,azTicks,elTicks,Rinterp);
        else; Rinterp=viewSynthesisLin(aList,eList,rList,azTicks,elTicks,Rinterp); end
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

% Fills a triangular face using planar interpolation (R = a*Az + b*El + c)
% In: aList, eList, rList [1x3] vertices; azTicks, elTicks; curView [nEl x nAz]
% Out: curView [nEl x nAz]
% Note: replaced by viewSynthesisNN in current benchmarks
function curView = viewSynthesisLin(aList,eList,rList,azTicks,elTicks,curView)
    A=[aList' eList' ones(3,1)]; planeCoef=A\rList';
    minA=min(aList); maxA=max(aList); minE=min(eList); maxE=max(eList);
    azTickInd=find(azTicks>=minA&azTicks<=maxA);
    elTickInd=find(elTicks>=minE&elTicks<=maxE);
    [locA,locE]=meshgrid(azTicks(azTickInd),elTicks(elTickInd));
    inMask=inTriangle(aList,eList,locA,locE);
    locR=Inf(size(locA));
    locR(inMask)=locA(inMask)*planeCoef(1)+locE(inMask)*planeCoef(2)+planeCoef(3);
    if ~isempty(elTickInd)&&~isempty(azTickInd)
        curView(elTickInd,azTickInd)=min(curView(elTickInd,azTickInd),locR);
    end
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