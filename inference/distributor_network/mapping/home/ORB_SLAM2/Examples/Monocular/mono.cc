/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/core/mat.hpp>
#include"../../include/System.h"
#include"../../include/MapPoint.h"
#include <semaphore.h>
#include <fcntl.h>

#include <math.h>

using namespace std;

class ImageGrabber
{
public:
    
    sem_t *sem; 
    int image_fifo;
    int frame_counter;
    char* data;
    int size;
    
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM)
    {   
        frame_counter = 0;
        //sem_open ("name", flags, permissions, start value)
        sem = sem_open ("/sem", O_CREAT | O_EXCL, 0644, 0);
        
        image_fifo = open("./image_fifo", O_RDONLY);
        if (image_fifo == -1) {
            perror("open failed");
        }
        size = 720 * 960;
        data = new char [size];
    }

    void GrabImage();

    ORB_SLAM2::System* mpSLAM;
};

int main(int argc, char **argv)
{
    
    if(argc != 3)
    {
        cerr << endl << "Usage: ./Mono path_to_vocabulary path_to_settings" << endl;        
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    ImageGrabber igb(&SLAM);
    if (igb.image_fifo == -1){
        return 1;
    }

    while(true)
    {
        igb.GrabImage();
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    
    vector< ORB_SLAM2::MapPoint * > allMapPoints = SLAM.GetMap()->GetAllMapPoints();
    ofstream plyfile("points.ply");

    vector< ORB_SLAM2::MapPoint * > goodPoints;

    for (int i = 0; i < allMapPoints.size(); i++)
    {
        if (!allMapPoints[i]->isBad())
        {
            goodPoints.push_back(allMapPoints[i]);
        }

    }
    
    if(plyfile.is_open())
    {
	    plyfile << "ply" << endl;
	    plyfile << "format ascii 1.0" << endl;
	    plyfile << "element vertex " << goodPoints.size() << endl;
	    plyfile << "property double x" << endl;
	    plyfile << "property double y" << endl;
	    plyfile << "property double z" << endl;
	    plyfile << "end_header" << endl;
	    for(int i = 0; i < goodPoints.size(); i++)
	    {
	        cv::Mat pos = goodPoints[i]->GetWorldPos();
	        plyfile << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
	    }
            plyfile.close();
    }


    return 0;
}

void ImageGrabber::GrabImage()
{
    read(image_fifo, data, size);
    //cv::Mat image = cv::Mat(width, height, CV_8UC4, (unsigned*)data);
    cv::Mat image = cv::Mat(720, 960, CV_8UC1, data);
    mpSLAM->TrackMonocular(image, 0, frame_counter);
    frame_counter++;
}


