Collection of Scripts to Read in and process data from a Microsoft Kinect v1 either online from the NYU-dataset or from own data acquisition
for the purpose of creating a new dataset for the training of Neural Nets to estimate Depth from RGB-Images.

Currently, there are two different Scripts to start the dataset creation: One for processing data from the online-dataset NYU and the oner one
for processing data from own data acquisition with the Kinect.The reason for that is that the file formats from the two different input sources
were different and it was the quickiest way to create these two different Scripts which are nearly identical.
Of course, this is not a clean solution and has to be refined in a later.

########### First starter Script ##################
The first starter Script is the DataProcessing_NYUData.py. It relies on the URL-list specified in the ListOfURLS.py which are links
to the NYU-indoor dataset: ttps://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

The URL-list must be kept up to date. It uses the NYUCameraParams.py, which is filled by the parameters which we found on the website of
the NYU dataset.

You also need to specify an OUTPUT-DIRECTORY for processed data. This will later become the input data for training.

This Script performs the Alignment between the depth Image of the Kinect with the corresponding RGB-Image (both in 640x480).
The RGB Images are undistorted in this process and saved in its original shape of 640x480 while the aligned Depth-Image is downsampled to 320x240
and saved as a grey scale image. 

It further invokes the PlaneSegmentationAndLabeling.py to eventually create a Binary-Mask as a PNG-file on which every white point represents
a 3D-point in the PCL of the depth Image of a Kinect frame which relates to a Plane.
It further creates a .npy-File which holds the actual PCL.

########### Second starter Script ##################
The second starter Script is nearly identical to the first one. The only difference is that it doesnt load files from the internet.
Instead it operates directly on the data directory created by some data acquisition process.
For speed acceleration this task is distributed on 4 indepentend processes using pythons multiprocessing API.
This speeds up the processing time up to 4 times but requires that you seperate the original input directory into 4 seperate ones.

The resultings paths have to be adjusted in the script.

Another difference is that it import the CameraParams.py instead of the NYUCameraParams.py. Here we had inserted parameters which we
got from a Kinect-Calibration under Linux with the Stereo Calibration Toolset from the MRPT-Software.

Everything else remains identicial to the first Starter script, including the necessity to specify the output directory.
