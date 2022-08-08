#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

int main() {
    Mat img = imread("D:/Git/NewCuda/unitTests/resources/ff14Me.png");
    Size dsize = Size(round(0.3 * img.cols), round(0.3 * img.rows));
    resize(img, img, dsize, 0, 0, INTER_LINEAR);
    imshow("ff14Me", img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}