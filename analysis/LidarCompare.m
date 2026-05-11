    %%% Setup - Set user parameters (PRM)
    %  Identify point clouds - refence and current
    prm.refFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';
    prm.refFrameNum = 1;
    prm.newFileName = 'C:\Users\wilso\RifeResearch\VolpeOuster_0318_1059_1500.mat';%
    prm.newFrameNum = 11; % Total 60
    
    %  lidar model description
    prm.azStart = 2*pi;              % Nominal start angle for each frame (rad)
    prm.azDir = -1;                  % Sign of azimuth change (positive for increasing angle; negative for decreasing angle)
    prm.minEl =-11.27*pi/180;        % Min elevation of scan (rad)
    prm.maxEl = 10.63*pi/180;        % Max elevation of scan (rad)
    % interpolation 
    prm.interpMethod = 0;            % Options: 0 is no interpolation, 1 is nearest neighobr, 2 is planar fit
    
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
    newAER = vec_xyz2aer(newXYZraw,prm);
    newXYZreg = prm.R*double(newXYZraw(1:3,:))+...
        repmat(prm.delta,[1 size(newXYZraw,2)]);                        % Apply pose transform to register current image to reference
    newAERreg = vec_xyz2aer(newXYZreg,prm);
    newAERreg(3,prm.domainMaskColIndxAER) = -1;                         % Exclude raypaths describing ego vehicle, setting range to -1 >> negative ranges should be impossible
    newAERinterpNN = vec_aerInterpMeth(newAERreg,prm);                    % Convert XYZ to AER array (ref pose coord) on regular grid, performing view synthesis
    % % newAERinterp = ndft(newAERreg,prm);
    % % newAERinterp = aerFFTInterp(newXYZraw,prm);
    % % Reference image
    refAER = vec_xyz2aer(refXYZ,prm);
    refAER(3,prm.domainMaskColIndxAER) = -1;                            % Exclude raypaths describing ego vehicle, setting range to -1 >> negative ranges should be impossible
    refAERinterp = vec_aerInterpMeth(refAER,prm);                       % Interpolate AER for reference image, noting Az coordinates are *not* constant over elevation
    % % newAERinterp = ndft(refAER,prm);
    newAERinterp = fft2lookupreverse(newAER, refAER,prm);
    newAERinterp(3,prm.domainMaskColIndxAER) = -1;                         % Exclude raypaths describing ego vehicle, setting range to -1 >> negative ranges should be impossible

    % newAERinterp = nufftInterp(newAERreg,prm);
    % newAERinterp = AERnurecon(newAERreg, prm);
    
    
    % %%% FFT Reconstruction
    % miniAERinterp = miniFFTinterp(newXYZraw,prm);
    
    % % refAER(3,(prm.nEl-1)*prm.nAz+1 : prm.nEl*prm.nAz) = 256;
    % refAERmask = maskAER(refAER);
    % refAERrecon = AER(refAER,prm),prm);
    % refAERrecon = refAERrecon .* refAERmask;
    %%% Visualizations
    %  
    figure(1); clf; %3D point clouds 
    plot3(refXYZ(1,:),refXYZ(2,:),refXYZ(3,:),'b.',newXYZreg(1,:),newXYZreg(2,:),newXYZreg(3,:),'r.');
    title('Two Lidar Point Clouds (after registration)')
    xlabel('X-Distance from reference lidar location (m)');
    ylabel('Y-Distance (m)')
    zlabel('Z-Distance (m)')
    

    figure(2); clf; %Range images
    sgtitle('LiDAR Range Images - View Synthesis Pipeline','FontSize', 14, 'FontWeight', 'bold');

    rawGrid = Inf(prm.nEl, prm.nAz);
    Az = newAERreg(1,:);
    El = newAERreg(2,:);
    R  = newAERreg(3,:);
    valid = isfinite(R) & R > 0 & R < 200;
    rowIdx = round((prm.maxEl - El(valid)) ./ (prm.maxEl - prm.minEl) .* (prm.nEl-1)) + 1;
    colIdx = round((Az(valid) - prm.azStart) ./ (prm.azDir * 2*pi) .* prm.nAz) + 1;
    rowIdx = max(1, min(prm.nEl, rowIdx));
    colIdx = mod(colIdx-1, prm.nAz) + 1;
    linIdx = sub2ind([prm.nEl prm.nAz], rowIdx, colIdx);
    rawGrid(linIdx) = R(valid);
    el_grid = linspace(prm.maxEl, prm.minEl, prm.nEl);
    az_grid = prm.azStart + prm.azDir * (0:prm.nAz-1) / prm.nAz * 2*pi;
    [AzMat, ElMat] = meshgrid(az_grid, el_grid);
    nCols = prm.nEl * prm.nAz;
    
    rawAER = [reshape(AzMat,   1, nCols);
              reshape(ElMat,   1, nCols);
              reshape(rawGrid, 1, nCols)];

    subplot(4,1,1);
    vizAzEl(rawAER, prm);
    title('Current Frame at Reference Origin - Before Interpolation')

    subplot(4,1,2); 
    vizAzEl(newAERinterp,prm);
    title('Interpolated at Current Time (FFT Lookup)')

    subplot(4,1,3); vizAzEl(refAERinterp,prm);
    % ylabel('Elevation angle (pixel count out of 128)')
    title('Reference Frame')
    newR = newAERinterp(3,:);
    newR(find(newR<0 | isinf(newR))) = NaN;
    newR = reshape(newR,[prm.nEl,prm.nAz]);
    refR = refAERinterp(3,:);
    refR(find(refR<0 | isinf(refR))) = NaN;
    refR = reshape(refR,[prm.nEl,prm.nAz]);
    dR = newR-refR;
    dR(isinf(dR) | isnan(dR)) = 0;
    disp(sqrt(mean(mean(dR.^2))));
    % subplot(5,1,4);
    % imagesc(abs(dR)*200, [0 255]); 
    % % ylabel('Elevation angle (pixel count out of 128)')
    % title('Range Difference (Current minus Reference)')
    subplot(4,1,4);
    imagesc(abs(dR));
    title('Absolute Range Difference per Pixel (Current minus Reference) (meters)')
    han = axes(gcf, 'Visible', 'off'); 
    han.YLabel.Visible = 'on';
    ylabel(han, 'Elevation angle (pixel count out of 128)');
    han.XLabel.Visible = 'on';
    xlabel(han, 'Azimuth angle (pixel count out of 2048)');
    % xlabel()
    
     figure(3); clf;
    binWidth = 0.4;
    edges = 0 : binWidth : 250;
    
    % filter out zeros from masked pixels before plotting
    dR_abs = abs(dR(:));
    dR_abs = dR_abs(dR_abs > 1e-6);   % exclude masked zeros
    
    histogram(dR_abs, edges, 'Normalization', 'probability','DisplayName','Absolute Pixel Range Error '); hold on;
    
    lambda = 1 / mean(dR_abs);
    x = linspace(0, 250, 10000);
    exp_pdf = lambda * exp(-lambda * x);
    plot(x, exp_pdf * binWidth, 'r-', 'LineWidth', 3, 'DisplayName', ...
        sprintf('Exponential PDF (\\lambda=%.3f)', lambda));
    
    set(gca, 'YScale', 'log');
    set(gca, 'XScale', 'log');
    xlabel('Absolute Range Error (m)');
    ylabel('Proportion');
    title('Distribution of Interpolation Error');
    xlim([binWidth 200]);
    legend('FontSize',15);
    grid on;


    
    % Draw FFT
    drawfft(refAER,prm);




    %%%%%% HELPER FUNCTIONS
    
    

    function newAERinterp = fft2lookupreverse(newAER, refAER, prm)
        
        % 1. Build FFT from new frame
        newR = reshape(newAER(3,:), prm.nEl, prm.nAz);
        invalid = ~isfinite(newR) | newR <= 0;
        newMean = mean(newR(~invalid));
        newR(invalid) = newMean;
        newR0 = newR - newMean;
        F_new = fft2(newR0);
        [N, M] = size(F_new);
        Fvec = F_new(:);
        [Krow, Kcol] = ndgrid(0:N-1, 0:M-1);
        Krow = Krow(:);
        Kcol = Kcol(:);
        
        % 2. Transform reference points into new frame
        R_inv     = prm.R';
        delta_inv = -prm.R' * prm.delta;
        
        validRef = isfinite(refAER(3,:)) & refAER(3,:) > 0 & refAER(3,:) < 200;
        validIdx = find(validRef);
        
        refXYZ_valid = vec_aer2xyz(refAER(:, validRef));
        xyz_newframe = R_inv * refXYZ_valid + repmat(delta_inv, 1, size(refXYZ_valid, 2));
        newFrameAER  = vec_xyz2aer(xyz_newframe, prm);
        
        % 3. Fractional indices into new frame FFT
        AzN = newFrameAER(1,:);
        ElN = newFrameAER(2,:);
        
        u = (prm.maxEl - ElN) ./ (prm.maxEl - prm.minEl) .* (N - 1);
        v = (AzN - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
        u = mod(u, N);
        v = mod(v, M);
        
        % 4. Chunked fractional IDFT — synthesized range is in new frame
        nPts = length(AzN);
        rangeNewFrame = zeros(1, nPts);
        CHUNK = 1024;
        
        for i = 1:CHUNK:nPts
            idx = i:min(i+CHUNK-1, nPts);
            u_c = u(idx);
            v_c = v(idx);
            phase = 2*pi * (Krow .* (u_c/N) + Kcol .* (v_c/M));
            rangeNewFrame(idx) = real(Fvec.' * exp(1j * phase)) / (N*M);
        end
        
        rangeNewFrame = rangeNewFrame + newMean;
        
        % 5. Convert synthesized range to reference frame range
        % Use synthesized range + new frame angles -> XYZ -> ref frame -> range
        synthAER_newframe = [AzN; ElN; rangeNewFrame];
        xyz_synth_new = vec_aer2xyz(synthAER_newframe);
        xyz_synth_ref = prm.R * xyz_synth_new + repmat(prm.delta, 1, nPts);
        
        % Range in reference frame = distance from reference origin
        rangeRefFrame = sqrt(sum(xyz_synth_ref.^2, 1));
        
        % 6. Assign directly to known reference pixel — no scatter
        % validIdx already encodes exact reference grid position for each point
        rangeRefFrame(rangeRefFrame < 0.5 | rangeRefFrame > 200) = Inf;
        
        rangeGrid = Inf(N, M);
        rangeGrid(validIdx) = rangeRefFrame;
        
        % 7. Output on reference grid
        el_grid = linspace(prm.maxEl, prm.minEl, N);
        az_grid = prm.azStart + prm.azDir * (0:M-1) / M * 2*pi;
        [AzOut, ElOut] = meshgrid(az_grid, el_grid);
        nCols = N * M;
        
        newAERinterp = [reshape(AzOut,     1, nCols);
                        reshape(ElOut,     1, nCols);
                        reshape(rangeGrid, 1, nCols)];
    end
    
    
    function newAERinterp = nufftInterp(newAERreg, prm)
    
        oversample = 4;
        W    = 6;
        beta = 2.34 * W;
        
        N  = prm.nEl;
        M  = prm.nAz;
        No = N * oversample;
        Mo = M * oversample;
        
        elTicks = linspace(prm.maxEl, prm.minEl, No);
        azTicks = prm.azStart + prm.azDir * (0:Mo-1) / Mo * 2*pi;
        
        Az = newAERreg(1,:);
        El = newAERreg(2,:);
        R  = newAERreg(3,:);
        
        valid = isfinite(R) & R > 0 & R < 200;
        AzV = Az(valid);
        ElV = El(valid);
        RV  = R(valid);
        P   = length(RV);
        
        refMean = mean(RV);
        RV = RV - refMean;
        
        u = (prm.maxEl - ElV) ./ (prm.maxEl - prm.minEl) .* (No - 1);
        v = (AzV - prm.azStart) ./ (prm.azDir * 2*pi) .* Mo;
        u = mod(u, No);
        v = mod(v, Mo);
        
        kb = @(x) real(besseli(0, beta * sqrt(max(1 - (2*x/W).^2, 0)))) / besseli(0, beta);
        
        [du, dv] = ndgrid(-W:W, -W:W);
        du = du(:)';
        dv = dv(:)';
        
        gridR = zeros(No, Mo);
        gridW = zeros(No, Mo);
        
        u0 = round(u);
        v0 = round(v);
        
        for k = 1:P
            uu  = mod(u0(k) + du, No) + 1;
            vv  = mod(v0(k) + dv, Mo) + 1;
            dx  = u(k) - (u0(k) + du);
            dy  = v(k) - (v0(k) + dv);
            w   = kb(dx) .* kb(dy);
            lin = sub2ind([No Mo], uu, vv);
            gridR(lin) = gridR(lin) + w * RV(k);
            gridW(lin) = gridW(lin) + w;
        end
        
        hasData = gridW > 0;
        Rinterp = zeros(No, Mo);
        Rinterp(hasData) = gridR(hasData) ./ gridW(hasData);
        
        Mask = double(hasData);
        [uaz, uel] = meshgrid(linspace(-0.5, 0.5, Mo), linspace(-0.5, 0.5, No));
        sigma = 0.4;
        Gwin  = ifftshift(exp(-(uaz.^2 + uel.^2) / (2*sigma^2)));
        LP_RM = real(ifft2(fft2(Rinterp .* Mask) .* Gwin));
        LP_M  = real(ifft2(fft2(Mask)            .* Gwin));
        
        Rsmooth = zeros(No, Mo);
        enough  = LP_M > 0.001;
        Rsmooth(enough) = LP_RM(enough) ./ LP_M(enough);
        
        Rinterp(~hasData & enough)  = Rsmooth(~hasData & enough);
        Rinterp(~hasData & ~enough) = 0;
        Rinterp(Rinterp < 0) = 0;
        
        Fo = fft2(Rinterp);
        
        Fc = zeros(N, M);
        Fc(1:N/2,   1:M/2)     = Fo(1:N/2,        1:M/2);
        Fc(1:N/2,   M/2+1:M)   = Fo(1:N/2,        Mo-M/2+1:Mo);
        Fc(N/2+1:N, 1:M/2)     = Fo(No-N/2+1:No,  1:M/2);
        Fc(N/2+1:N, M/2+1:M)   = Fo(No-N/2+1:No,  Mo-M/2+1:Mo);
        
        kb1d_row = zeros(No, 1);
        kb1d_col = zeros(Mo, 1);
        for i = -W:W
            kb1d_row(mod(i, No) + 1) = kb(i);
            kb1d_col(mod(i, Mo) + 1) = kb(i);
        end
        KB_row = abs(fft(kb1d_row));
        KB_col = abs(fft(kb1d_col));
        
        KB_row_c = [KB_row(1:N/2);   KB_row(No-N/2+1:No)];
        KB_col_c = [KB_col(1:M/2);   KB_col(Mo-M/2+1:Mo)];
        deapod   = KB_row_c * KB_col_c';
        deapod(deapod < 1e-3 * max(deapod(:))) = 1e-3 * max(deapod(:));
        
        Fc = Fc ./ deapod;
        
        Rout = real(ifft2(Fc));
        Rout = Rout + refMean;
        Rout(Rout < 0.5 | Rout > 200) = Inf;
        
        rowIdx = round(linspace(1, No, N));
        colIdx = round(linspace(1, Mo, M));
        Rout   = Rout(rowIdx, colIdx);
        
        el_grid = elTicks(rowIdx);
        az_grid = azTicks(colIdx);
        [AzOut, ElOut] = meshgrid(az_grid, el_grid);
        nCols = N * M;
        
        newAERinterp = [reshape(AzOut, 1, nCols);
                        reshape(ElOut, 1, nCols);
                        reshape(Rout,  1, nCols)];
    end
    
    
    
    
    
    
    function drawfft(refAER,prm)
        AERarray = refAER(3,:);
        AERarray(~isfinite(AERarray) | AERarray <= 0) = 0;
        AERarray = reshape(AERarray,prm.nEl,prm.nAz);
        F = (fft2(AERarray));
        Fmag = 20*log10(abs(F)+eps);
        
        fAz = linspace(0,1, prm.nAz);
        fEl = linspace(0,1, prm.nEl);
    
        figure(4); clf;
        subplot(2,1,1);
        vizAzEl(refAER,prm);
        title('Reference Scan');
    
        subplot(2,1,2);
        imagesc(fAz, fEl, Fmag);
        colorbar;
        xlabel('Normalized Azimuth Frequency'); ylabel('Normalized Elevation Frequency');
        title('2D FFT Magnitude (dB)');
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
                
        upsample  = 4;
        kernelW   = 2;      % half-width in oversampled grid cells — increase to reduce holes
        sigma_splat = 2;  % Gaussian splat width in grid cells
        
        nElDec = prm.nEl * upsample;
        nAzDec = prm.nAz * upsample;
        elTicks = linspace(prm.maxEl, prm.minEl, nElDec);
        azTicks = prm.azStart + prm.azDir*(0:nAzDec-1)/nAzDec*2*pi;
        
        curR = listAER(3,:);
        valid = curR > 0 & isfinite(curR) & curR < 200;
        az = listAER(1,valid);
        el = listAER(2,valid);
        curR = curR(valid);
        P = length(curR);
        
        aPos = (az - azTicks(1)) / (azTicks(2) - azTicks(1)) + 1;
        ePos = (el - elTicks(1)) / (elTicks(2) - elTicks(1)) + 1;
        
        [dk_e, dk_a] = ndgrid(-kernelW:kernelW, -kernelW:kernelW);
        dk_e = dk_e(:)';   % [1 x K]
        dk_a = dk_a(:)';
        
        Rinterp = zeros(nElDec, nAzDec);
        Winterp = zeros(nElDec, nAzDec);
        
        e0 = round(ePos);
        a0 = round(aPos);
        
        for k = 1:P
            ee  = e0(k) + dk_e;
            aa  = a0(k) + dk_a;
            
            % wrap azimuth, clamp elevation
            aa  = mod(aa - 1, nAzDec) + 1;
            valid_e = ee >= 1 & ee <= nElDec;
            ee  = max(1, min(nElDec, ee));
            
            w   = exp(-(dk_e.^2 + dk_a.^2) / (2 * sigma_splat^2));
            w(~valid_e) = 0;
            
            lin = sub2ind([nElDec nAzDec], ee, aa);
            Rinterp(lin) = Rinterp(lin) + w * curR(k);
            Winterp(lin) = Winterp(lin) + w;
        end
        
        hasData = Winterp > 0;
        Rinterp(hasData)  = Rinterp(hasData) ./ Winterp(hasData);
        Rinterp(~hasData) = 0;
        
        M_mask = double(hasData);
        [uaz, uel] = meshgrid(linspace(-0.5,0.5,nAzDec), linspace(-0.5,0.5,nElDec));
        sigma_smooth = 0.2;
        Gwin  = ifftshift(exp(-(uaz.^2 + uel.^2) / (2*sigma_smooth^2)));
        LP_RM = real(ifft2(fft2(Rinterp .* M_mask) .* Gwin));
        LP_M  = real(ifft2(fft2(M_mask)            .* Gwin));
        
        Rsmooth = zeros(nElDec, nAzDec);
        enough  = LP_M > 0.001;
        Rsmooth(enough) = LP_RM(enough) ./ LP_M(enough);
        
        Rinterp(~hasData & enough)  = Rsmooth(~hasData & enough);
        Rinterp(~hasData & ~enough) = 0;
        Rinterp(Rinterp < 0) = 0;
        
        rowIdx = round(linspace(1, nElDec, prm.nEl));
        colIdx = round(linspace(1, nAzDec, prm.nAz));
        Rout = Rinterp(rowIdx, colIdx);
        Rout(Rout <= 0 | Rout > 200) = Inf;
        
        [AzGrid, ElGrid] = meshgrid(azTicks(colIdx), elTicks(rowIdx));
        nCols = prm.nEl * prm.nAz;
        newAERinterp = [reshape(AzGrid,1,nCols); ...
                        reshape(ElGrid,1,nCols); ...
                        reshape(Rout,  1,nCols)];
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
    
    
    
    %%% VIZUALIZATION
    function vizAzEl(imAER,prm)
        [Amat,Emat,Rmat] = vec2mataer(imAER,[prm.nEl size(imAER,2)/prm.nEl]); 
        image(3./Rmat*255*2);
    end
