#include "pano2perspective.hh"

void create_intrinsic_param(cv::Mat_<float> &K,
    const int w,
    const int h,
    const float fov)
{
    // 1:1 (pixel) aspect ratio is assumed:
    float fov_rad = fov * M_PI / 180.0;
    float f = (float)w/(2. * tan(fov_rad / 2.));
    K = (
        cv::Mat_<float>(3, 3) << 
        f,  0., (float)w/2,
        0., f,  (float)h/2,
        0., 0., 1.);
}

void angle2Rot(const array<float, 3> &rot_angle, cv::Mat_<float> &rot)
{
    // Coordinate space: Y facing down, X forward
    cv::Mat_<float> A = (cv::Mat_<float>(3,3) << 
		1., 0., 0.,
		0., cos(rot_angle[1]), -sin(rot_angle[1]),
		0., sin(rot_angle[1]), cos(rot_angle[1]));
    
    cv::Mat_<float> B = (cv::Mat_<float>(3,3) <<
		cos(rot_angle[0]), 0., -sin(rot_angle[0]),
		0., 1., 0.,
		sin(rot_angle[0]), 0., cos(rot_angle[0]));

    cv::Mat_<float> C = (cv::Mat_<float>(3,3) <<
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
	const float &u,
	const float &v,
	cv::Vec3b &pixel)
{
    float v0 = (1.0 - (v-(int)v))*(1.0 - (u-(int)u));
    float v1 = ((v-(int)v))*(1.0 - (u-(int)u));
    float v2 = (1.0 - (v-(int)v))*((u-(int)u));
    float v3 = ((v-(int)v))*((u-(int)u));
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
            cv::Mat imgpt = (cv::Mat_<float>(3,1) << (float)j, (float)i, 1.0);
            cv::Mat ori = T_im2ori * imgpt;
            float D = sqrt(
                ori.at<float>(0,0) * ori.at<float>(0,0) +
                ori.at<float>(1,0) * ori.at<float>(1,0) +
                ori.at<float>(2,0) * ori.at<float>(2,0));
            float phi = asin(ori.at<float>(1,0) / D); // [-pi/2:pi/2]
            float theta = atan2(ori.at<float>(0,0), ori.at<float>(2,0)); // [-pi:pi]

            // z-axis = front direction in the panorama image
	        float u = (theta + M_PI) * (pano.cols/(2.0*M_PI)) - .5;
	        float v = (phi + M_PI/2) * (pano.rows/(M_PI)) - .5;
            bilinear_interpolation(pano, u, v, pers.at<cv::Vec3b>(i,j));
        }
    }
}

cv::Mat process_single_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src,
    array<float, 3> &angles,
    bool cuda,
    int w, int h,
    float fov)
{

    // Convert Eigen Matrix of the input image to cv::Mat
    cv::Mat pano = cv::Mat::zeros(src[0].rows(), src[0].cols(), CV_8UC3);
    //cv::Mat rgb[3];
    //eigen2mat(src, rgb);
    //cv::merge(rgb, 3, pano);
    vector<cv::Mat> rgb;

    cv::Mat r = cv::Mat(src[0].rows(), src[0].cols(), CV_8UC1, src[0].data());
    cv::Mat g = cv::Mat(src[1].rows(), src[1].cols(), CV_8UC1, src[1].data());
    cv::Mat b = cv::Mat(src[2].rows(), src[2].cols(), CV_8UC1, src[2].data());
    
    rgb.push_back(b);
    rgb.push_back(g);
    rgb.push_back(r);
    cv::merge(rgb, pano);

    // Set size of perspective image
    cv::Size size_pers = cv::Size(w, h);
    cv::Mat pers = cv::Mat::zeros(size_pers, CV_8UC3);

    // Get intrinsic and rotation matrix
    cv::Mat_<float> K;
    cv::Mat_<float> rot;
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
    int w, int h, float fov)
{ 
    init_values(w, h, fov);
}

Pano2Perspective::Pano2Perspective(
    int pano_w, int pano_h, float fov_x, float fov_y)
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
)
{
    //cv::Mat pano = cv::Mat::zeros(src[0].rows(), src[0].cols(), CV_8UC3);
    pano = cv::Mat::zeros(src[0].rows(), src[0].cols(), CV_8UC3);
    
    //cv::Mat rgb[3];
    //eigen2mat(src, rgb);
    //cv::merge(rgb, 3, pano);
    vector<cv::Mat> rgb;

    cv::Mat r = cv::Mat(src[0].rows(), src[0].cols(), CV_8UC1, src[0].data());
    cv::Mat g = cv::Mat(src[1].rows(), src[1].cols(), CV_8UC1, src[1].data());
    cv::Mat b = cv::Mat(src[2].rows(), src[2].cols(), CV_8UC1, src[2].data());
    
    rgb.push_back(b);
    rgb.push_back(g);
    rgb.push_back(r);

    cv::merge(rgb, pano);

    // Create perspective image
    //cv::Mat pers = cv::Mat::zeros(size_pers, CV_8UC3);

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

array<float, 4> Pano2Perspective::get_intrinsics()
{
    float fx = K.at<float>(0,0);
    float fy = K.at<float>(1,1);
    float x0 = K.at<float>(0,2);
    float x1 = K.at<float>(1,2);
    array<float, 4> intrinsics = {fx, fy, x0, x1};
    return intrinsics;
}

void Pano2Perspective::init_values(int w, int h, float fov)
{
    size_pers = cv::Size(w, h);
    
    // init pers
    pers = cv::Mat::zeros(size_pers, CV_8UC3);

    create_intrinsic_param(K, w, h, fov);
    
    array<float, 3> angles = {0.0, 0.0, 0.0};
    set_rotation(angles);
}

void Pano2Perspective::set_rotation(array<float, 3> &angles)
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

    float w = size_pano.width;
    float h = size_pano.height;
    float yaw = (x - w/2) * 2*M_PI / w;
    float pitch = (y - h/2) * M_PI / h;

    array<float, 3> angles = {yaw, pitch, 0};
    angle2Rot(angles, rot);
    T_im2ori = rot.inv() * K.inv();
}

void Pano2Perspective::cuda(int id)
{
    // cout << "Using CUDA" << endl;
    CUDA = true;
    device = id;
}
