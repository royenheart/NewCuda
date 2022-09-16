#ifndef __IMAGE__
#define __IMAGE__

#include <iostream>
#include <string>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void* big_random_block( int size ) {
    unsigned char *data = (unsigned char*)malloc( size );
    for (int i=0; i<size; i++)
        data[i] = rand();

    return data;
}

typedef struct DataBlock DataBlock;
typedef struct BoolBlock BoolBlock;

template<typename rawImageT>
class IMAGE {
public:
    Mat image;
    String name;
    int sizePerElement;

    IMAGE(int w, int h) {
        image = Mat::zeros(w,h,CV_8UC4);
        sizePerElement = sizeof(rawImageT);
    }

    // 使用委托构造函数（不能直接构造函数内使用其他构造函数，并不会对当前对象进行修改，c++11的特性）
    IMAGE(int w, int h, String myname): IMAGE(w, h) {
        name = myname;
    }

    unsigned char* get_ptr( void ) const   
    { 
        return (unsigned char*)image.data; 
    }

    /**
     * @brief 显示图像大小（弃用）
     * @deprecated
     * 
     * @return long image size
     */
    long image_size( void ) const 
    { 
		return image.cols * image.rows * sizePerElement;
    }

    /**
     * @brief 返回原始图像大小
     * 
     * @return long 
     */
    long rawImage_size( void ) const 
    { 
		return image.cols * image.rows * sizePerElement;
    }

    /**
     * @brief 返回结果图像大小
     * 
     * @return long 
     */
    long resultImage_size( void ) const 
    { 
		return image.cols * image.rows * 4;
    }

    char show_image(int time=0)
    {
        imshow(this->name, image);
        return waitKey(time);
    }

    template<typename T, typename block>
    int show_animations(T opt, block *d, int ticks) {
        while (true) {
            // Do the image operation first
            opt(d, ticks);
            imshow(name, image);
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
    IMAGE<float> *bitmap;
    
    cudaEvent_t start, stop;
    float totalTime;
    int frams;
};

struct BoolBlock {
    unsigned char *output_bitmap;
    bool *dev_outSrc;
    bool *dev_inSrc;
    IMAGE<bool> *bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    int frams;
};

#endif  // __IMAGE__

