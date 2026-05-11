 cd 'C:\Users\wilso\RifeResearch'

clear ousterReader;
ousterReader = ousterFileReader(pcapFile, jsonFile);
shiftStartTime = ousterReader.Timestamps(shiftStartFrame + 1);
ousterReader.CurrentTime = shiftStartTime;

% User parameters
userSelectData = 1;   % Default is 1; change to 2 for dual-return data
flagPlayMovie = false;
flagWriteClip = true;
writeClipNumber = 11; % If flagWriteClip is true, which clip to write?

% Read Data
shiftStartFrame = 1;  
switch userSelectData
    case 1
        % First (of many) data files from 2025 Zeta collection - tour about
        %  Fairfax suburbs
        pcapFile = "OS-2-128_v2.5.3_2048x10_20250318_105927-000.pcap";
        jsonFile = "OS-2-128_v2.5.3_2048x10_20250318_105927.json";
        if flagWriteClip
            clipLen = 60;
            stationaryFrames = [10 1100 1850 8900 11350]; %Times when lidar stationary 
            motionFrames = [1500 2200 3500 5000 5700 6500 7500 8500 10500] ;
            listStartFrames = [stationaryFrames motionFrames];
            shiftStartFrame = listStartFrames(writeClipNumber);  
        end
        clipHead = 'VolpeOuster_0318_1059_';
end

% Read Data
if ~exist("ousterReader")
    ousterReader = ousterFileReader(pcapFile, jsonFile);
end

% View LIDAR movie
if flagPlayMovie
    player = pcplayer([-60 60],[-60 60],[-30 30]);
    shiftStartTime = ousterReader.Timestamps(shiftStartFrame+1);
    ousterReader.CurrentTime = shiftStartTime; % Reset point cloud "current time"
    %if flagWriteClip; finalFrame = clipLen; else finalFrame = ousterReader.NumberOfFrames-shiftStartFrame; end
    for n=1:finalFrame
        n+shiftStartFrame
        if mod(n+shiftStartFrame,500)==0; pause; end % Pause every 500 time steps
        ptCloudObj = readFrame(ousterReader);
        rangeSq = sum(ptCloudObj(1).Location.^2,3);
        [tooCloseIndxI,tooCloseIndxJ] = find(rangeSq < 1^2);
        view(player,ptCloudObj(1).Location,ptCloudObj(1).Intensity);
        pause(0.01);

        % Extract Data for file write operation
        if flagWriteClip
            if n>0 & n<=clipLen
                pc{n}=ptCloudObj(1).Location;
                frameNum(n) = n+shiftStartFrame;
                reflect{n} = ptCloudObj(1).Intensity;
                if userSelectData == 2 % If sampling dual return data
                    pc2nd{n}=ptCloudObj(2).Location;
                    reflect2nd{n} = ptCloudObj(2).Intensity;
                end
            end
        end
    end
end

% Write data to file
clipName = [clipHead num2str(shiftStartFrame)];
if flagWriteClip & userSelectData == 2
     eval(['save ' clipName ' pc pc2nd frameNum reflect reflect2nd']);
elseif flagWriteClip & userSelectData == 1
     eval(['save ' clipName ' pc frameNum reflect']);    
end
