%%%%%%%%%%%%%%%  Compute mean model and errors 

clear;
close all;

% go to directory with stable points and this matlab code
cd '/tilde/fleet/reports/appearModels/journal/DataAnalysis/stablePoints/'

FALSE = (0 == 1);
TRUE = ~FALSE;

SANITY_CHECK = FALSE;

frame = 0:1144;
nMousePnts = 7;

% directory name that contains individual frames of Dudek Sequence
dataDir = '/tilde/fleet/data/sequences/dudekface1/';


%%%%%%%%%%%%% Read results of all moused in points in original coords
mousePnts = [];
for sequenceType = 1:nMousePnts   
  fnData =['origPointCoord_' num2strPad(sequenceType, 1) '.dat']
  fid = fopen(fnData, 'r');
  pnts = reshape(fscanf(fid,'%f %f\n'),2,length(frame));
  fclose(fid);
  mousePnts = [mousePnts; pnts];
end
mousePnts = reshape(mousePnts,[2,nMousePnts,length(frame)]);


%%%%%%%%%%%%% Plot moused in points in original coords
for k = 1:5:length(frame)
    fnum = frame(k);
    fname = [dataDir, 'frame', num2strPad(fnum, 4), '.pgm'];
    fprintf(1,'frame %d: ',fnum);
    im = pgmRead(fname);
    
    oPnts = mousePnts(:,:,k);
    idx = find(oPnts(1,:) > 0);
    clf; displayImage(im); hold on; truesize;
    plot(oPnts(1,idx), oPnts(2,idx),'*r');
    
    pause(0.1);
end

