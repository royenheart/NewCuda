#ifndef __IMAGE__
#define __IMAGE__

#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

typedef struct DataBlock DataBlock;

class IMAGE {
public:
    Mat image;

    IMAGE( int w, int h) 
    {
        image = Mat::zeros(w,h,CV_8UC4);
    }

    unsigned char* get_ptr( void ) const   
    { 
        return (unsigned char*)image.data; 
    }

    long image_size( void ) const 
    { 
		return image.cols * image.rows * 4; 
    }

    char show_image(int time=0)
    {
        imshow("images", image);
        return waitKey(time);
    }

    int show_animations(void (*opt)(DataBlock*, int), DataBlock *d, int ticks) {
        while (true) {
            // Do the image operation first
            opt(d, ticks);
            imshow("image", image);
            if (waitKey(100) == 27) {
                destroyAllWindows();
                break;
            }
        }
        return 0;
    }
};

struct DataBlock {
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    IMAGE *bitmap;
    
    cudaEvent_t start, stop;
    float totalTime;
    float frams;
};

#endif  // __IMAGE__

