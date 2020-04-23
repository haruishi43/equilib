#include "pano2perspective.hh"

void create_intrinsic_param(cv::Mat_<double> &K,
    const int w,
    const int h,
    const double fov)
{
    // 1:1 (pixel) aspect ratio is assumed:
    double fov_rad = fov * M_PI / 180.0;
    double f = (double)w/(2. * tan(fov_rad / 2.));
    K = (
        cv::Mat_<double>(3, 3) << 
        f,  0., (double)w/2,
        0., f,  (double)h/2,
        0., 0., 1.);
}

void angle2Rot(const array<double, 3> &rot_angle, cv::Mat_<double> &rot)
{
    // Coordinate space: Y facing down, X forward
    cv::Mat_<double> A = (cv::Mat_<double>(3,3) << 
		1., 0., 0.,
		0., cos(rot_angle[1]), -sin(rot_angle[1]),
		0., sin(rot_angle[1]), cos(rot_angle[1]));
    
    cv::Mat_<double> B = (cv::Mat_<double>(3,3) <<
		cos(rot_angle[0]), 0., -sin(rot_angle[0]),
		0., 1., 0.,
		sin(rot_angle[0]), 0., cos(rot_angle[0]));

    cv::Mat_<double> C = (cv::Mat_<double>(3,3) <<
        cos(rot_angle[2]), sin(rot_angle[2]), 	0.,
		-sin(rot_angle[2]), cos(rot_angle[2]), 	0.,
		0., 0., 1.);

	rot = A * B * C;
}

void eigen2mat(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    cv::Mat (&out)[3])
{
    const size_t nthreads = src.size();
    vector<thread> threads(nthreads);
    for(int t=0; t<nthreads; t++)
    {
        threads[t] = thread(bind(
            [&](const int i, const int ei, const int t)
            {
                cv::Mat channel(src[i].rows(), src[i].cols(), CV_8UC1, src[i].data());
                out[i] = channel;
            }, t, t, t));
    }
    std::for_each(threads.begin(), threads.end(), [](std::thread& x){ x.join(); });
}

void bilinear_interpolation(
	const cv::Mat &src,
	const double &u,
	const double &v,
	cv::Vec3b &pixel)
{
    double v0 = (1.0 - (v-(int)v))*(1.0 - (u-(int)u));
    double v1 = ((v-(int)v))*(1.0 - (u-(int)u));
    double v2 = (1.0 - (v-(int)v))*((u-(int)u));
    double v3 = ((v-(int)v))*((u-(int)u));
	pixel[0] = src.at<cv::Vec3b>((int)v, (int)u)[0] * v0
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[0] * v1
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[0] * v2
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[0] * v3;
    pixel[1] = src.at<cv::Vec3b>((int)v, (int)u)[1] * v0
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[1] * v1
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[1] * v2
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[1] * v3;
    pixel[2] = src.at<cv::Vec3b>((int)v, (int)u)[2] * v0
		+ src.at<cv::Vec3b>((int)v+1, (int)u)[2] * v1
		+ src.at<cv::Vec3b>((int)v, (int)u+1)[2] * v2
		+ src.at<cv::Vec3b>((int)v+1, (int)u+1)[2] * v3;
}

void crop(
    const cv::Mat &pano,
    cv::Mat &pers,
    const cv::Mat &T_im2ori)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) 
#endif
    for(int i=0; i<pers.rows; i++)
    {
        for(int j=0; j<pers.cols; j++)
        {
            cv::Mat imgpt = (cv::Mat_<double>(3,1) << (double)j, (double)i, 1.0);
            cv::Mat ori = T_im2ori * imgpt;
            double D = sqrt(
                ori.at<double>(0,0) * ori.at<double>(0,0) +
                ori.at<double>(1,0) * ori.at<double>(1,0) +
                ori.at<double>(2,0) * ori.at<double>(2,0));
            double phi = asin(ori.at<double>(1,0) / D); // [-pi/2:pi/2]
            double theta = atan2(ori.at<double>(0,0), ori.at<double>(2,0)); // [-pi:pi]

            // z-axis = front direction in the panorama image
	        double u = (theta + M_PI) * (pano.cols/(2.0*M_PI)) - .5;
	        double v = (phi + M_PI/2) * (pano.rows/(M_PI)) - .5;
            bilinear_interpolation(pano, u, v, pers.at<cv::Vec3b>(i,j));
        }
    }
}

cv::Mat process_single_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    array<double, 3> &angles,
    bool cuda,
    int w, int h,
    double fov)
{
    // Convert Eigen Matrix of the input image to cv::Mat
    cv::Mat pano(src[0].rows(), src[0].cols(), CV_8UC3);
    cv::Mat rgb[3];
    eigen2mat(src, rgb);
    cv::merge(rgb, 3, pano);

    // Set size of perspective image
    cv::Size size_pers = cv::Size(w, h);
    cv::Mat pers = cv::Mat::zeros(size_pers, CV_8UC3);

    // Get intrinsic and rotation matrix
    cv::Mat_<double> K;
    cv::Mat_<double> rot;
    create_intrinsic_param(K, w, h, fov);
    angle2Rot(angles, rot);

    // Calculate orientation matrix
    cv::Mat T_im2ori = rot.inv() * K.inv();
    
    if (cuda)
    {
        //FIXME: temp device
        int device = 0;
        // Create perspective image (CUDA)
        crop_gpu(pano, pers, T_im2ori, device);
    }
    else
    {
        // Create perspective image (serial/OpenMP)
        crop(pano, pers, T_im2ori);
    }

    return pers;
}

Pano2Perspective::Pano2Perspective(
    int w, int h, double fov)
{ 
    init_values(w, h, fov);
}

Pano2Perspective::Pano2Perspective(
    int pano_w, int pano_h, double fov_x, double fov_y)
{
    size_pano = cv::Size(pano_w, pano_h);
    int w = (int)(pano_w * fov_x / 360.0 + 0.5); // round to integer
    int h = (int)(pano_h * fov_y / 180.0 + 0.5);
    init_values(w, h, fov_x);
}

Pano2Perspective::~Pano2Perspective()
{

}

cv::Mat Pano2Perspective::process_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src
) const
{
    cv::Mat pano(src[0].rows(), src[0].cols(), CV_8UC3);
    cv::Mat rgb[3];
    eigen2mat(src, rgb);
    cv::merge(rgb, 3, pano);

    // Create perspective image
    cv::Mat pers = cv::Mat::zeros(size_pers, CV_8UC3);

    // Crop Panorama 
    if (CUDA)
    {
        crop_gpu(pano, pers, T_im2ori, device);
    }
    else
    {
        crop(pano, pers, T_im2ori);
    }

    return pers;
}

array<double, 4> Pano2Perspective::get_intrinsics()
{
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double x0 = K.at<double>(0,2);
    double x1 = K.at<double>(1,2);
    array<double, 4> intrinsics = {fx, fy, x0, x1};
    return intrinsics;
}

void Pano2Perspective::init_values(int w, int h, double fov)
{
    size_pers = cv::Size(w, h);
    create_intrinsic_param(K, w, h, fov);
    array<double, 3> angles = {0.0, 0.0, 0.0};
    set_rotation(angles);
}

void Pano2Perspective::set_rotation(array<double, 3> &angles)
{
    if (K.empty())
    {
        cout << "Intrinsic parameters are not initialized" << endl;
        return;
    }
    angle2Rot(angles, rot);
    T_im2ori = rot.inv() * K.inv();
}

void Pano2Perspective::set_center_point(int x, int y)
{
    if (size_pano.width == 0 || size_pano.height == 0)
    {
        cout << "Panorama size is not initialized" << endl;
        return;
    }

    double w = size_pano.width;
    double h = size_pano.height;
    double yaw = (x - w/2) * 2*M_PI / w;
    double pitch = (y - h/2) * M_PI / h;

    array<double, 3> angles = {yaw, pitch, 0};
    angle2Rot(angles, rot);
    T_im2ori = rot.inv() * K.inv();
}

void Pano2Perspective::cuda(int id)
{
    cout << "Using CUDA" << endl;
    CUDA = true;
    device = id;
}
