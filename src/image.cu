/**
 * @file image.cpp
 * @author royenheart (royenheart@outlook.com)
 * @brief 图像显示用
 * @version 0.1
 * @date 2023-01-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "../inc/image.cuh"
#include "../inc/block.cuh"
#include <cstdlib>

template<typename rawImageT>
IMAGE<rawImageT>::IMAGE(int w, int h) {
    image = Mat::zeros(w, h, CV_8UC4);
    sizePerElement = sizeof(rawImageT);
}

template<typename rawImageT>
IMAGE<rawImageT>::IMAGE(int w, int h, String myname): IMAGE(w, h) {
    name = myname;
}

template<typename rawImageT>
unsigned char* IMAGE<rawImageT>::get_ptr(void) const {
    return (unsigned char*)image.data; 
}

template<typename rawImageT>
long IMAGE<rawImageT>::image_size(void) const {
    return image.cols * image.rows * sizePerElement;
}

template<typename rawImageT>
long IMAGE<rawImageT>::rawImage_size(void) const {
    return image.cols * image.rows * sizePerElement;
} 

template<typename rawImageT>
long IMAGE<rawImageT>::resultImage_size(void) const {
    return image.cols * image.rows * 4;
}

template<typename rawImageT>
char IMAGE<rawImageT>::show_image(int time) {
    imshow(this->name, image);
    return waitKey(time);
} 

template<typename rawImageT>
template<typename T, typename block>
int IMAGE<rawImageT>::show_animations(T opt, block *d, int ticks) {
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

// 显式实例化

template class IMAGE<float>;
template int IMAGE<float>::show_animations(void (*opt)(BoolBlock*, int), BoolBlock *d, int ticks);
template int IMAGE<float>::show_animations(void (*opt)(DataBlock*, int), DataBlock *d, int ticks);
template class IMAGE<bool>;
template int IMAGE<bool>::show_animations(void (*opt)(BoolBlock*, int), BoolBlock *d, int ticks);
template int IMAGE<bool>::show_animations(void (*opt)(DataBlock*, int), DataBlock *d, int ticks);