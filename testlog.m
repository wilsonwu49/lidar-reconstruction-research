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
