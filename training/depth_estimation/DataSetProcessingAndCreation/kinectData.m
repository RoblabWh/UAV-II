kinectDataFolderPath = 'E:/Downloads2/kinectDataEcht2/pack10';
kinectDataIndexPath = fullfile(kinectDataFolderPath, 'INDEX.txt');
fileId = fopen(kinectDataIndexPath,"w");
cVid = videoinput('kinect', 1);
dVid = videoinput('kinect', 2);

cVid.FramesPerTrigger = 1;
dVid.FramesPerTrigger = 1;

framesLimit = 150;
frameStepsToSave = 60;
triggerCountLimit = framesLimit * frameStepsToSave;
cVid.TriggerRepeat = triggerCountLimit;
dVid.TriggerRepeat = triggerCountLimit;

triggerconfig([cVid dVid],'manual');

start([cVid dVid]);

for i = 1:triggerCountLimit
    trigger([cVid dVid])
    [imgColor, ts_color, metaData_Color] = getdata(cVid);
    [imgDepth, ts_depth, metaData_Depth] = getdata(dVid);
    m = mod(i, frameStepsToSave);
    if m==0
        tC = metaData_Color.AbsTime;
        tD = metaData_Depth.AbsTime;
        
        
        imgColorName = sprintf('%d%d%d%d%d%f.ppm',tC);
        imgDepthName = sprintf('%d%d%d%d%d%f.pgm',tD);
        
        pathToPpm = fullfile(kinectDataFolderPath, imgColorName);
        pathToPgm = fullfile(kinectDataFolderPath, imgDepthName);
        
        indexLineColor = sprintf("%s\n", imgColorName);
        indexLineDepth = sprintf("%s\n", imgDepthName);
        
        fprintf(fileId,indexLineColor);
        fprintf(fileId,indexLineDepth);
        
        imwrite(imgColor, pathToPpm);
        imwrite(imgDepth, pathToPgm);
        
    end
end
        
fclose(fileId);
delete([cVid dVid]);