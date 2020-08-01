#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Unaligned;

#pragma once

/*
 * Create instrinsic parameter given the size of the image and fov (for width)
 */
void create_intrinsic_param(cv::Mat_<double> &K,
    const int w,
    const int h,
    const double fov);

/*
 * Get rotation matrix from the direction given
 * - Input: an array of double (3 rotation angles)
 * - Output: rotation matrix
 */
void angle2Rot(const array<double, 3> &rot_angle, cv::Mat_<double> &rot);

/*
 * Convert Eigen matrix with 3 channels to cv::Mat with 3 channels
 * Used 3 threads for each channels.
 * OpenMP did not work well for this.
 */
void eigen2mat(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    cv::Mat (&out)[3]);


/*
 * Binlinear Interpolation
 */
void bilinear_interpolation(
	const cv::Mat &src,
	const double &u,
	const double &v,
	cv::Vec3b &pixel);

/*
 * Main function to convert panorama image to perspective image
 */
void crop(
    const cv::Mat &pano,
    cv::Mat &pers,
    const cv::Mat &T_im2ori);

/*
 * Main function to convert panorama image to perspective image using GPU
 */
void crop_gpu(
    const cv::Mat &pano,
    cv::Mat &pers,
    const cv::Mat &T_im2ori,
    int device);

/*
 * Wrapper function for cropping a perspective image from the panorama image.
 * 
 * Input:
 * - panorama ([R, G, B])
 * - rotation ([yaw, pitch, roll])
 * - (opt) cuda boolean
 * - (opt) width int
 * - (opt) height int
 * 
 * Output:
 * - perspective (a custom `Image` class)
 * 
 */
cv::Mat process_single_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    array<double, 3> &angles,
    bool cuda=false,
    int w=480, int h=360,
    double fov=90.0);

/*
 * Wrapper class when cropping multiple images in a row (video)
 */ 
class Pano2Perspective
{
public:

    // Initialize using perspective image size and fov (in deg)
    Pano2Perspective(int w=480, int h=360, double fov=90.0);
    // Initialize using panorama image size and fov (x,y in deg)
    Pano2Perspective(int pano_w, int pano_h, double fov_x, double fov_y);
    ~Pano2Perspective();

    // Getter
    cv::Mat process_image(array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src) const;
    array<double, 4> get_intrinsics();

    // Setter
    void init_values(int w, int h, double fov);
    void set_rotation(array<double, 3> &angles);
    void set_center_point(int x, int y);
    void cuda(int id=0);

private:

    // GPU flag
    bool CUDA = false;
    // Device ID
    int device;
    // Size of perspective (width, height)
    cv::Size size_pers = cv::Size(0, 0);
    // Size of panorama (width, height)
    cv::Size size_pano = cv::Size(0, 0);
    // Instrinsic Parameter
    cv::Mat_<double> K;
    // 3x3 Rotation Matrix
    cv::Mat_<double> rot;
    // 3x3 orientation Matrix
    cv::Mat T_im2ori;
};
