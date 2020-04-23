
#include <iostream>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>

#include "pano2perspective.hh"


using namespace std;

int main(int, char**)
{

    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    
    cap.open("./data/test.mp4");

    if (!cap.isOpend())
    {
        cerr << "ERROR! Unable to open file\n";
        return -1;
    }

    
    //--- INITIALIZE pano2perspective
    int width = 640;
    int height = 480;
    double fov = 90.0;
    Pano2Perspective p2p(width, height, fov);
    
    int device = 0;
    p2p.cuda(device);


    
    //--- START LOOP
    cout << "Start video" << endl;

    for (;;)
    {
        // wait for a new frame from video and store it to 'frame'
        cap.read(frame);

        if (frame.empty())
        {
            cerr << "ERROR! blank frame\n";
            break;
        }
        
    }

    return 0;
}
