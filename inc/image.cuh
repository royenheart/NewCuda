#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include <string>

template<typename rawImageT>
class IMAGE {
public:
    Mat image;
    String name;
    int sizePerElement;

    IMAGE(int w, int h); 

    // 使用委托构造函数（不能直接构造函数内使用其他构造函数，并不会对当前对象进行修改，c++11的特性）
    IMAGE(int w, int h, String myname);

    unsigned char* get_ptr(void) const; 

    /**
     * @brief 显示图像大小（弃用）
     * @deprecated
     * 
     * @return long image size
     */
    long image_size(void) const;

    /**
     * @brief 返回原始图像大小
     * 
     * @return long 
     */
    long rawImage_size(void) const; 

    /**
     * @brief 返回处理过的图像大小
     * 
     * @return long 
     */
    long resultImage_size(void) const; 

    char show_image(int time=0); 

    template<typename T, typename block>
    int show_animations(T opt, block *d, int ticks);
};