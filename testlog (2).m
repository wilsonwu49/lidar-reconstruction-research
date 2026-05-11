function newAERinterp = 1(newAERreg, refAER, prm)

    elStart = 1; elEnd = 128;
    azStart = 1601; azEnd = 1728;
    
    H = elEnd - elStart + 1;
    W = azEnd - azStart + 1;
    
    el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    az = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    
    elChunk = el(elStart:elEnd);
    azChunk = az(azStart:azEnd);
    
    % ----- reference chunk -----
    reflistR = refAER(3,:);
    reflistR(isnan(reflistR)) = 0;
    
    refGridFull = reshape(reflistR, prm.nEl, prm.nAz);
    refGrid = refGridFull(elStart:elEnd , azStart:azEnd);
    
    refFFT = fft2(refGrid);
    
    % ----- new frame chunk -----
    newAz = reshape(newAERreg(1,:), prm.nEl, prm.nAz);
    newEl = reshape(newAERreg(2,:), prm.nEl, prm.nAz);
    
    newChunkAz = newAz(elStart:elEnd , azStart:azEnd);
    newChunkEl = newEl(elStart:elEnd , azStart:azEnd);
    
    newChunkAz = newChunkAz(:);
    newChunkEl = newChunkEl(:);
    
    N = numel(newChunkAz);
    values = zeros(N,1);
    
    for i = 1:N
    
        x = (newChunkAz(i)-azChunk(1))/(azChunk(end)-azChunk(1))*(W-1);
        y = (newChunkEl(i)-elChunk(1))/(elChunk(end)-elChunk(1))*(H-1);
    
        val = 0;
    
        for k = 0:H-1
            for l = 0:W-1
                val = val + refFFT(k+1,l+1) * exp(1j*2*pi*(k*y/H + l*x/W));
            end
        end
    
        values(i) = real(val)/(H*W);
    
    end
    
    output = NaN(prm.nEl, prm.nAz);
    output(elStart:elEnd , azStart:azEnd) = reshape(values,H,W);
    
    [AzGrid, ElGrid] = meshgrid(az, el);
    
    nCols = prm.nEl * prm.nAz;
    
    newAERinterp = [reshape(AzGrid,1,nCols);
                    reshape(ElGrid,1,nCols);
                    reshape(output,1,nCols)];

end



function newAERinterp = 2(newAERreg, refAER, prm)

    elStart = 97; elEnd = 128;
    azStart = 1601; azEnd = 1632;
    
    H = elEnd - elStart + 1;
    W = azEnd - azStart + 1;

    reflistR = refAER(3,:);
    reflistR(isnan(reflistR)) = 0;
    refGrid = reshape(reflistR, prm.nEl, prm.nAz);
    avgRef = mean(refGrid(:));
    refGrid = refGrid - avgRef;
    refAERfft = fft2(refGrid);

% % Image Test refGrid and IFFT - Correct
% test = real(ifft2(refAERfft));
% figure(5);
% subplot(3,1,1);
% imagesc(test);
% subplot(3,1,2);
% imagesc(refGrid);
% dR = test - refGrid;
% subplot(3,1,3);
% imagesc(dR);

    newAz = reshape(newAERreg(1,:), prm.nEl, prm.nAz);
    newEl = reshape(newAERreg(2,:), prm.nEl, prm.nAz);

    newChunkAz = newAz(elStart:elEnd , azStart:azEnd);
    newChunkEl = newEl(elStart:elEnd , azStart:azEnd);

    query_row = (newChunkEl(:) - prm.minEl) / (maxEl - minEl) * (H-1);
    query_col = (newChunkAz(:)) / (maxAz - minAz) * (W-1);
    query = [query_row(:), query_col(:)];
    output = real(nufftn(refGrid,[],query)) + avgRef;
    output = reshape(output,H,W);

    el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    az = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    r = zeros(prm.nEl, prm.nAz);
    r(elStart:elEnd, azStart:azEnd) = output;
    [AzGrid, ElGrid] = meshgrid(az, el);

    nCols = prm.nEl * prm.nAz;

    newAERinterp = [reshape(AzGrid,1,nCols);
                    reshape(ElGrid,1,nCols);
                    reshape(r,1,nCols)];

end





function newAERinterp = ndft(listAER, prm)

    % ----- Chunk selection -----
    elStart = 1; elEnd = 128;
    azStart = 1600; azEnd = 2000;

    H = elEnd - elStart + 1;
    W = azEnd - azStart + 1;

    el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    az = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;

    elChunk = el(elStart:elEnd);
    azChunk = az(azStart:azEnd);

    % ----- Extract AER lists -----
    listAz = listAER(1,:);
    listEl = listAER(2,:);
    listR  = listAER(3,:);

    valid = listR > 0 & isfinite(listR) ...
          & listEl >= min(elChunk) & listEl <= max(elChunk) ...
          & listAz >= min(azChunk) & listAz <= max(azChunk);

    chunkAz = listAz(valid)';
    chunkEl = listEl(valid)';
    chunkR  = listR(valid)';

    N = length(chunkR);
    disp(N);
    disp(N/(H*W));

    % ----- Remove mean -----
    meanR = mean(chunkR);
    chunkR = chunkR - meanR;

    % ----- Normalize coordinates to grid -----
    t_el = (chunkEl - elChunk(1)) / (elChunk(end)-elChunk(1)) * H;
    t_az = (chunkAz - azChunk(1)) / (azChunk(end)-azChunk(1)) * W;

    % ----- Frequency indices -----
    pIdx = -floor(H/2):ceil(H/2)-1;
    qIdx = -floor(W/2):ceil(W/2)-1;

    % ----- Build NDFT matrix -----
    K = H*W;
    A = zeros(N, K);

    idx = 1;
    for ip = 1:H
        p = pIdx(ip);
        for iq = 1:W
            q = qIdx(iq);
            A(:,idx) = exp(2*pi*1i*(p*t_el/H + q*t_az/W));
            idx = idx + 1;
        end
    end

    % ----- Solve least-squares system -----
    lambda = 1e-3;  % regularization
    c = (A'*A + lambda*eye(K)) \ (A'*chunkR);

    % ----- Reshape Fourier coefficients -----
    C = reshape(c, H, W);

    % ----- Reconstruct grid via inverse Fourier series -----
    Rchunk = zeros(H, W);

    for m = 0:H-1
        for n = 0:W-1
            val = 0;
            for ip = 1:H
                p = pIdx(ip);
                for iq = 1:W
                    q = qIdx(iq);
                    val = val + C(ip,iq) * exp(2*pi*1i*(p*m/H + q*n/W));
                end
            end
            Rchunk(m+1, n+1) = real(val);
        end
    end

    % ----- Add mean back -----
    Rchunk = Rchunk + meanR;
    Rchunk(Rchunk <= 0 | Rchunk > 200) = NaN;

    % ----- Package output -----
    Rout = NaN(prm.nEl, prm.nAz);
    Rout(elStart:elEnd, azStart:azEnd) = Rchunk;

    [AzGrid, ElGrid] = meshgrid(az, el);

    nCols = prm.nEl * prm.nAz;
    newAERinterp = [reshape(AzGrid,1,nCols);
                    reshape(ElGrid,1,nCols);
                    reshape(Rout,1,nCols)];

end

function newAERinterp = nufftInterp(listAER, prm)
% NUFFT (Type-1) + IDFT
% Converts nonuniform spatial points to uniform grid
%
% INPUT:
%   listAER: [Az; El; R]  (nonuniform measurements)
%   prm: structure with fields
%       prm.nEl     - number of elevation points (Nx)
%       prm.nAz     - number of azimuth points (Ny)
%       prm.maxEl   - max elevation
%       prm.minEl   - min elevation
%       prm.azStart - start azimuth
%       prm.azDir   - azimuth direction (scale)
%
% OUTPUT:
%   newAERinterp: [Az; El; Rgrid] on uniform grid

    % Extract data
    Az = listAER(1,:);
    El = listAER(2,:);
    R  = listAER(3,:);
    
    % Keep valid points only
    valid = R > 0 & isfinite(R);
    Az = Az(valid);
    El = El(valid);
    R = R(valid);
    
    % NUFFT: nonuniform -> uniform frequency grid
    % Define 2D nonuniform sample locations
    t = [El(:), Az(:)];  % Nx2 matrix
    
    % Compute NUFFT (type-1)
    Yf = nufftn(R(:), t);  % MATLAB automatically assumes type-1 if query points omitted
    
    
    % Inverse FFT to get uniform spatial grid
    Rgrid = real(ifft2(Yf));
    
    % Build uniform Az/El grids
    elGrid = linspace(prm.maxEl, prm.minEl, prm.nEl);
    azGrid = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    [AzG, ElG] = meshgrid(azGrid, elGrid);
    
    % Return as flat arrays
    newAERinterp = [reshape(AzG,1,prm.nEl * prm.nAz);
                    reshape(ElG,1,prm.nEl * prm.nAz);
                    reshape(Rgrid,1,prm.nEl * prm.nAz)];
end



function newAERinterp = miniFFTinterp(newXYZraw, prm)
    % Convert to AER
    AERvec = vec_xyz2aer(newXYZraw, prm);
    AERvec(1,:) = mod(AERvec(1,:) - prm.azStart, 2*pi) + prm.azStart;
    invalid = isnan(AERvec(3,:)) | isinf(AERvec(3,:)) | (AERvec(3,:) < 0);
    AERvec(3,invalid) = 0;

    % Reshape to grid
    AERarray = reshape(AERvec(3,:), prm.nEl, prm.nAz);

    % Define target grid
    az = linspace(prm.azStart, prm.azStart + prm.azDir*2*pi, prm.nAz);
    el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    [AzGrid, ElGrid] = meshgrid(az, el);

    % Convert target grid to XYZ in old frame
    targetAER = [AzGrid(:)'; ElGrid(:)'; 100*ones(1, numel(AzGrid))];
    targetXYZ = vec_aer2xyz(targetAER);
    targetXYZ_old = prm.R' * (double(targetXYZ) - prm.delta);

    % Convert back to AER
    newAERvec = vec_xyz2aer(targetXYZ_old, prm);
    newAERvec(1,:) = mod(newAERvec(1,:) - prm.azStart, 2*pi) + prm.azStart;

    % FFT resampling
    F = fft2(AERarray);
    Nx = prm.nAz; Ny = prm.nEl;
    uIdx = (newAERvec(1,:) - az(1)) / (az(end) - az(1)) * (Nx-1);
    vIdx = (el(1) - newAERvec(2,:)) / (el(1) - el(end)) * (Ny-1);
    uIdx = max(min(uIdx, Nx-1), 0);
    vIdx = max(min(vIdx, Ny-1), 0);
    k = 0:Ny-1;
    l = 0:Nx-1;

    % Preallocate output
    Rout = zeros(1, numel(uIdx));
    
    for p = 1:numel(uIdx)
        expRow = exp(1j*2*pi*k*vIdx(p)/Ny);   % row term
        expCol = exp(1j*2*pi*l*uIdx(p)/Nx);   % column term
        Rout(p) = sum(sum(F .* (expRow.' * expCol))) / (Nx*Ny);
    end
    
    Rout = real(Rout); 

    % Assemble interpolated AER
    newAERinterp = [newAERvec(1,:); newAERvec(2,:); Rout(:)'];
end

function newAERinterp = aerFFTInterp(newXYZraw,prm)
    % Convert old grid 
    AERvec = vec_xyz2aer(newXYZraw,prm);
    AERvec(1,:) = mod(AERvec(1,:) - prm.azStart, 2*pi) + prm.azStart;
    invalid = isnan(AERvec(3,:)) | isinf(AERvec(3,:)) | (AERvec(3,:) < 0);
    AERvec(3,invalid) = 256;
    AERarray = reshape(AERvec(3,:),prm.nEl,prm.nAz);
    AERarray(isnan(AERarray) | isinf(AERarray) | AERarray < 0) = 0;

    % Create new grid
    az = linspace(prm.azStart, prm.azStart+prm.azDir*2*pi, prm.nAz);
    el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    [AzNew, ElNew] = meshgrid(az, el);
    AERvec2 = [AzNew(:)'; ElNew(:)'; ones(1,numel(AzNew))];

    % Calculate points for new grid on old coordinate system
    XYZvec2 = vec_aer2xyz(AERvec2);
    XYZvec2new = prm.R' * (double(XYZvec2) - prm.delta);
    newAERvec = vec_xyz2aer(XYZvec2new,prm);
    newAERvec(1,:) = mod(newAERvec(1,:) - prm.azStart, 2*pi) + prm.azStart;

    % Calculate desired values from fft 
    Nx = prm.nAz;
    Ny = prm.nEl;
    azQueryIdx = (newAERvec(1,:) - prm.azStart) / (2*pi*prm.azDir) * (Nx-1);
    elQueryIdx = (prm.maxEl - newAERvec(2,:)) / (prm.maxEl - prm.minEl) * (Ny-1);
    ElNewIdx = (prm.maxEl - AERvec(2,:)) / (prm.maxEl - prm.minEl) * (Ny-1);
    AzNewIdx = (AERvec(1,:) - prm.azStart) / (2*pi*prm.azDir) * (Nx-1);
    azQueryIdx = max(min(azQueryIdx, Nx-1), 0);
    elQueryIdx = max(min(elQueryIdx, Ny-1), 0);
    AzNewIdx   = max(min(AzNewIdx,   Nx-1), 0);
    ElNewIdx   = max(min(ElNewIdx,   Ny-1), 0);
    f = [elQueryIdx(:), azQueryIdx(:)];
    t = [ElNewIdx(:),AzNewIdx(:)];
    validT = all(isfinite(t),2);
    validF = all(isfinite(f),2);
    t = t(validT,:);
    f = f(validF,:);


    % 2D NUFFT
    values = nufftn(AERarray, t,f);

    % 
    newAERvalues = [newAERvec(1,:);newAERvec(2,:);real(values(:)')];
    newXYZvalues = vec_aer2xyz(newAERvalues);
    newXYZvalues = prm.R*double(newXYZvalues(1:3,:))+repmat(prm.delta,[1 size(newXYZvalues,2)]);

    % Reshape back to grid
    newAERinterp = vec_xyz2aer(newXYZvalues,prm);
end


function newAERinterp = fft2lookup(newAERreg, refAER, prm)
    % 
    %     elStart = 65; elEnd = 128;
    %     azStart = 1601; azEnd = 1664;
    % 
    %     H = elEnd - elStart + 1;
    %     W = azEnd - azStart + 1;
    % 
    %     reflistR = refAER(3,:);
    %     reflistR(isnan(reflistR)) = 0;
    %     refGrid = reshape(reflistR, prm.nEl, prm.nAz);
    %     avgRef = mean(refGrid(:));
    %     refGrid = refGrid - avgRef;
    %     refAERfft = fft2(refGrid);
    % 
    % % Image Test refGrid and IFFT - Correct
    % test = real(ifft2(refAERfft));
    % figure(5);
    % subplot(3,1,1);
    % imagesc(test);
    % subplot(3,1,2);
    % imagesc(refGrid);
    % dR = test - refGrid;
    % subplot(3,1,3);
    % imagesc(dR);
    % newAERinterp = newAERreg;
    % 
        % CHUNK_SIZE = 1024;   % points per IDFT batch, tune to RAM
        % 
        % % Spatial chunk bounds (pixel indices into the full grid)
        % elStart = 1;  elEnd = 128;
        % azStart = 1; azEnd = 2048;
        % 
        % % Build reference grid
        % refR = refAER(3, :);
        % refR(isnan(refR)) = 0;
        % refGrid = reshape(refR, prm.nEl, prm.nAz);
        % avgRef = mean(refGrid(:));
        % refGrid = refGrid - avgRef;
        % 
        % % 2D FFT
        % F = fft2(refGrid);
        % [N, M] = size(F);
        % Fvec = F(:);
        % % Frequency index 
        % [Krow, Kcol] = ndgrid(0:N-1, 0:M-1);
        % Krow = Krow(:);
        % Kcol = Kcol(:);
        % 
        % % Az/El coordinate 
        % el = linspace(prm.maxEl, prm.minEl, prm.nEl);
        % az = prm.azStart + prm.azDir * (0:prm.nAz-1) / prm.nAz * 2*pi;
        % 
        % % Extract chunk bounds
        % elChunk = el(elStart:elEnd);
        % azChunk = az(azStart:azEnd);
        % H = length(elChunk);
        % W = length(azChunk);
        % 
        % % Query points for chunk
        % [AzQ, ElQ] = meshgrid(azChunk, elChunk);
        % AzQ = AzQ(:).';
        % ElQ = ElQ(:).';
        % 
        % % Fractional indices
        % u = (prm.maxEl - ElQ) ./ (prm.maxEl - prm.minEl) .* (N - 1);
        % v = (AzQ - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
        % u = mod(u, N);
        % v = mod(v, M);
        % 
        % % IDFT
        % nPts = H * W;
        % rangeChunk = NaN(1, nPts);
        % 
        % for start = 1:CHUNK_SIZE:nPts
        %     idx = start : min(start + CHUNK_SIZE - 1, nPts);
        %     u_c = u(idx);   % [1 x C]
        %     v_c = v(idx);
        % 
        %     % phase: [N*M x C]
        %     phase = 2*pi * (Krow .* (u_c/N) + Kcol .* (v_c/M));
        % 
        %     % IDFT synthesis: [1 x C]
        %     rangeChunk(idx) = real(Fvec.' * exp(1j * phase)) / (N*M);
        % end
        % 
        % rangeChunk = rangeChunk + avgRef;
        % rangeChunk(rangeChunk < 0.5) = Inf;
        % 
        % rFull = zeros(prm.nEl, prm.nAz);
        % rFull(elStart:elEnd, azStart:azEnd) = reshape(rangeChunk, H, W);
        % 
        % [AzGrid, ElGrid] = meshgrid(az, el);
        % nCols = prm.nEl * prm.nAz;
        % newAERinterp = [reshape(AzGrid, 1, nCols);
        %                 reshape(ElGrid, 1, nCols);
        %                 reshape(rFull,  1, nCols)];
    
        %     CHUNK_SIZE = 1024;
        % 
        % % Spatial chunk bounds (pixel indices into the full grid)
        % elStart = 1;   elEnd = 128;
        % azStart = 1;   azEnd = 2048;
        % 
        % %% Build reference range grid
        % refR = refAER(3, :);
        % refR(isnan(refR)) = 0;
        % refGrid = reshape(refR, prm.nEl, prm.nAz);
        % avgRef = mean(refGrid(:));
        % refGrid = refGrid - avgRef;
        % N = 128; M = 2048;
        % win = hann(N) * hann(M)'; 
        % refGrid = refGrid .* win;
        % 
        % %% 2D FFT
        % F = fft2(refGrid);
        % [N, M] = size(F);
        % Fvec = F(:);
        % 
        % %% Frequency index grids [N*M x 1]
        % [Krow, Kcol] = ndgrid(0:N-1, 0:M-1);
        % Krow = Krow(:);
        % Kcol = Kcol(:);
        % 
        % %% Az/El coordinate vectors for chunk bounds
        % el = linspace(prm.maxEl, prm.minEl, prm.nEl);
        % az = prm.azStart + prm.azDir * (0:prm.nAz-1) / prm.nAz * 2*pi;
        % elChunk = el(elStart:elEnd);
        % azChunk = az(azStart:azEnd);
        % 
        % %% Filter newAERreg to chunk bounds
        % newAz = newAERreg(1, :);
        % newEl = newAERreg(2, :);
        % 
        % inChunk = newEl >= min(elChunk) & newEl <= max(elChunk) & ...
        %           newAz >= min(azChunk) & newAz <= max(azChunk);
        % 
        % AzQ = newAz(inChunk);   % [1 x nChunkPts]
        % ElQ = newEl(inChunk);
        % 
        % %% Map to fractional 0-based grid indices
        % u = (prm.maxEl - ElQ) ./ (prm.maxEl - prm.minEl) .* (N - 1);
        % v = (AzQ - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
        % u = min(mod(u, N), N-1);   % clamp to avoid float boundary wrap
        % v = min(mod(v, M), M-1);
        % 
        % %% Chunked IDFT synthesis
        % nPts = length(AzQ);
        % rangeChunk = NaN(1, nPts);
        % 
        % for start = 1:CHUNK_SIZE:nPts
        %     idx = start : min(start + CHUNK_SIZE - 1, nPts);
        %     u_c = u(idx);
        %     v_c = v(idx);
        % 
        %     % phase: [N*M x C]
        %     phase = 2*pi * (Krow .* (u_c/N) + Kcol .* (v_c/M));
        % 
        %     % IDFT synthesis: [1 x C]
        %     rangeChunk(idx) = real(Fvec.' * exp(1j * phase)) / (N*M);
        % end
        % 
        % rangeChunk = rangeChunk + avgRef;
        % rangeChunk(rangeChunk < 0.5) = Inf;   % mask sky/leakage
        % 
        % %% Assemble output — keep newAERreg Az/El, replace range
        % newAERinterp = newAERreg;
        % newAERinterp(3, :) = Inf;              % default outside chunk
        % newAERinterp(3, inChunk) = rangeChunk;
    
    %     CHUNK_SIZE = 1024;   % points per IDFT batch
    % 
    % % ==========================================
    % % USER TEST: shift scene horizontally
    % % positive = right shift
    % % negative = left shift
    % % ==========================================
    % colShift = 25;      % try 5, 10, 25, 50, etc.
    % 
    % % Full grid bounds
    % elStart = 1;  
    % elEnd   = 128;
    % azStart = 1; 
    % azEnd   = 2048;
    % 
    % %% Build reference grid
    % refR = refAER(3,:);
    % refR(isnan(refR)) = 0;
    % 
    % refGrid = reshape(refR, prm.nEl, prm.nAz);
    % 
    % avgRef = mean(refGrid(:));
    % refGrid = refGrid - avgRef;
    % 
    % %% 2D FFT
    % F = fft2(refGrid);
    % [N,M] = size(F);
    % 
    % Fvec = F(:);
    % 
    % % Frequency indices
    % [Krow,Kcol] = ndgrid(0:N-1,0:M-1);
    % Krow = Krow(:);
    % Kcol = Kcol(:);
    % 
    % %% Angular coordinates
    % el = linspace(prm.maxEl, prm.minEl, prm.nEl);
    % az = prm.azStart + prm.azDir*(0:prm.nAz-1)/prm.nAz*2*pi;
    % 
    % %% Query grid = regular scene, shifted columns
    % elChunk = el(elStart:elEnd);
    % azChunk = az(azStart:azEnd);
    % 
    % [AzQ,ElQ] = meshgrid(azChunk, elChunk);
    % 
    % % ==========================================
    % % SHIFT columns by modifying azimuth query
    % % ==========================================
    % azStep = prm.azDir * 2*pi / prm.nAz;
    % AzQ = AzQ + colShift * azStep;
    % 
    % % wrap azimuth
    % AzQ = mod(AzQ, 2*pi);
    % 
    % AzQ = AzQ(:).';
    % ElQ = ElQ(:).';
    % 
    % H = length(elChunk);
    % W = length(azChunk);
    % 
    % %% Fractional indices
    % u = (prm.maxEl - ElQ) ./ (prm.maxEl - prm.minEl) .* (N-1);
    % v = (AzQ - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
    % 
    % u = mod(u,N);
    % v = mod(v,M);
    % 
    % %% IDFT evaluate shifted scene
    % nPts = H*W;
    % rangeChunk = NaN(1,nPts);
    % 
    % for start = 1:CHUNK_SIZE:nPts
    % 
    %     idx = start:min(start+CHUNK_SIZE-1,nPts);
    % 
    %     u_c = u(idx);
    %     v_c = v(idx);
    % 
    %     phase = 2*pi*(Krow.*(u_c/N) + Kcol.*(v_c/M));
    % 
    %     rangeChunk(idx) = real(Fvec.' * exp(1j*phase)) / (N*M);
    % end
    % 
    % %% Restore mean
    % rangeChunk = rangeChunk + avgRef;
    % rangeChunk(rangeChunk < 0.5) = Inf;
    % 
    % %% Assemble output
    % rFull = reshape(rangeChunk,H,W);
    % 
    % [AzGrid,ElGrid] = meshgrid(az,el);
    % 
    % nCols = prm.nEl * prm.nAz;
    % 
    % newAERinterp = [reshape(AzGrid,1,nCols);
    %                 reshape(ElGrid,1,nCols);
    %                 reshape(rFull ,1,nCols)];
    
    % refR = reshape(refAER(3,:), prm.nEl, prm.nAz);
    % 
    % refR = fillmissing(refR, 'nearest');
    % 
    % win = hann(prm.nEl) * hann(prm.nAz)';
    % refR = refR .* win;
    % 
    % F = fft2(refR);
    % 
    % [Krow, Kcol] = ndgrid(0:prm.nEl-1, 0:prm.nAz-1);
    % Krow = Krow(:);
    % Kcol = Kcol(:);
    % 
    % Fvec = F(:);
    % 
    % AzQ = newAERreg(1,:);
    % ElQ = newAERreg(2,:);
    % 
    % % correct index mapping
    % el_grid = linspace(prm.maxEl, prm.minEl, prm.nEl);
    % az_grid = linspace(0, 2*pi, prm.nAz);
    % 
    % u = interp1(el_grid, 1:prm.nEl, ElQ, 'linear', 'extrap');
    % v = interp1(az_grid, 1:prm.nAz, AzQ, 'linear', 'extrap');
    % 
    % nPts = length(u);
    % out = zeros(1,nPts);
    % 
    % CHUNK = 2048;
    % 
    % for i = 1:CHUNK:nPts
    %     idx = i:min(i+CHUNK-1,nPts);
    % 
    %     u_c = u(idx);
    %     v_c = v(idx);
    % 
    %     phase = exp(1j * 2*pi * ( ...
    %         Krow * (u_c./prm.nEl) + ...
    %         Kcol * (v_c./prm.nAz) ));
    % 
    %     out(idx) = real((Fvec.' * phase) / (prm.nEl*prm.nAz));
    % end
    % 
    % newAERinterp = newAERreg;
    % newAERinterp(3,:) = out;


%% USER PARAMETERS - spatial crop
elStart = 1;   elEnd = 128;
azStart = 1;   azEnd = 2048;

%% 1. Build full reference grid
refR = reshape(refAER(3,:), prm.nEl, prm.nAz);
invalid = ~isfinite(refR) | refR <= 0;
refMean = mean(refR(~invalid));
refR(invalid) = refMean;
refR0 = refR - refMean;

%% 2. FFT of full reference grid
F = fft2(refR0);
[N, M] = size(F);
Fvec = F(:);

[Krow, Kcol] = ndgrid(0:N-1, 0:M-1);
Krow = Krow(:);
Kcol = Kcol(:);

%% 3. Define angular bounds of crop
el_grid = linspace(prm.maxEl, prm.minEl, N);
az_grid = prm.azStart + prm.azDir * (0:M-1) / M * 2*pi;

elMin = min(el_grid(elStart), el_grid(elEnd));
elMax = max(el_grid(elStart), el_grid(elEnd));
azMin = min(az_grid(azStart), az_grid(azEnd));
azMax = max(az_grid(azStart), az_grid(azEnd));

%% 4. Filter transformed points to crop bounds + valid only
Az = newAERreg(1,:);
El = newAERreg(2,:);
R  = newAERreg(3,:);

inCrop = isfinite(R) & R > 0 & ...
         El >= elMin & El <= elMax & ...
         Az >= azMin & Az <= azMax;

AzV = Az(inCrop);
ElV = El(inCrop);

%% 5. Fractional 0-based indices
u = (prm.maxEl - ElV) ./ (prm.maxEl - prm.minEl) .* (N - 1);
v = (AzV - prm.azStart) ./ (prm.azDir * 2*pi) .* M;
u = mod(u, N);
v = mod(v, M);

%% 6. Chunked fractional IDFT
nPts = length(AzV);
rangeEval = zeros(1, nPts);
CHUNK = 1024;

for i = 1:CHUNK:nPts
    idx = i:min(i+CHUNK-1, nPts);
    u_c = u(idx);
    v_c = v(idx);
    phase = 2*pi * (Krow .* (u_c/N) + Kcol .* (v_c/M));
    rangeEval(idx) = real(Fvec.' * exp(1j * phase)) / (N*M);
end

rangeEval = rangeEval + refMean;

%% 7. Z-buffer into crop-sized subgrid
rowIdx = round(u) + 1;
colIdx = mod(round(v), M) + 1;
rowIdx = max(1, min(N, rowIdx));

% map to crop-local indices
rowLocal = rowIdx - elStart + 1;
colLocal = colIdx - azStart + 1;
H = elEnd - elStart + 1;
W = azEnd - azStart + 1;

inBounds = rowLocal >= 1 & rowLocal <= H & ...
           colLocal >= 1 & colLocal <= W;

rowLocal  = rowLocal(inBounds);
colLocal  = colLocal(inBounds);
rangeEval = rangeEval(inBounds);

cropGrid = Inf(H, W);
linIdx = sub2ind([H W], rowLocal, colLocal);
[rSort, si] = sort(rangeEval, 'descend');   % descend so minimum overwrites
cropGrid(linIdx(si)) = rSort;

%% 8. Insert crop into full output grid
rangeGrid = Inf(N, M);
rangeGrid(elStart:elEnd, azStart:azEnd) = cropGrid;

%% 9. Output on full reference grid
[AzOut, ElOut] = meshgrid(az_grid, el_grid);
nCols = N * M;

newAERinterp = [reshape(AzOut,     1, nCols);
                reshape(ElOut,     1, nCols);
                reshape(rangeGrid, 1, nCols)];
end