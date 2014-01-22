%
% This directory contains an image sequence called the Dudek sequence that
% was originally taken by Allan Jepson, David Fleet and Tom El-Maraghi at 
% the Palo Alto Research Center for experimental work on visual tracking.
%
% The image sequence is in DudekSeq/ .
% Ground truth data are in StablePoints/ .
% Below is some simple matlab code for displaying the ground truth
% point data on top of the original images from the sequence.
%


frame = 0:1144;
nMousePnts = 7;
dataDir = 'DudekSeq/frame';


%%%%%%%%%%%%%  Read results of ground truth pts in original coords %%%%%%%
mousePnts = [];
for sequenceType = 1:nMousePnts   
  fnData =['StablePoints/origPointCoord_' num2strPad(sequenceType, 1) '.dat']
  fid = fopen(fnData, 'r');
  pnts = reshape(fscanf(fid,'%f %f\n'),2,length(frame));
  fclose(fid);
  mousePnts = [mousePnts; pnts];
end
mousePnts = reshape(mousePnts,[2,nMousePnts,length(frame)]);


%%%%%%%%%%%%%  Plot ground truth pts on images in original coords  %%%%%%%%%
for k = 1:25:length(frame)
    fnum = frame(k);
    fname = [dataDir num2strPad(fnum, 4) '.pgm'];
    fprintf(1,'frame %d: ',fnum);
    im = pgmRead(fname);
    
    oPnts = mousePnts(:,:,k);
    idx = find(oPnts(1,:) > 0);
    clf; displayImage(im); hold on; 
    plot(oPnts(1,idx), oPnts(2,idx),'*r');
    pause(0.1);
end

