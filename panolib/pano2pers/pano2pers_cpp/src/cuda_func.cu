#include <sstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);                                
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)

/*
 Main function to convert panorama image to perspective image
 */
__global__ 
void crop_cuda(
    unsigned char *pano,
    unsigned char *pers,
    float *im2ori,
    int pano_w, int pano_h,
    int pers_w, int pers_h,
    int pano_step, int pers_step)
{
    // 2D Index of current thread
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only valid threads perform memory I/O 
	if((i < pers_w) && (j < pers_h))
    {
        // Location of colored pixel in output
        int pers_tid = j * pers_step + (3 * i);
        // Create orientation matrix
        float ori[3];
        for(int m=0; m<3; m++)
        {
            ori[m] = im2ori[m*3] * (float)i
                + im2ori[m*3+1] * (float)j
                + im2ori[m*3+2]; 
        }
        
        float D = sqrt(
            ori[0] * ori[0]
            + ori[1] * ori[1]
            + ori[2] * ori[2]);
        
        float phi = asin(ori[1] / D); // [-pi/2:pi/2]
        float theta = atan2(ori[0], ori[2]); // [-pi:pi]
        float u = (theta + M_PI) * (pano_w/(2.0*M_PI)) - .5;
        float v = (phi + M_PI/2) * (pano_h/(M_PI)) - .5;
        int px0 = (int)v*pano_step + 3*(int)u;
        int px1 = (int)(v+1)*pano_step + 3*(int)u;
        int px2 = (int)v*pano_step + 3*(int)(u+1);
        int px3 = (int)(v+1)*pano_step + 3*(int)(u+1);
        float v0 = (1.0-(v-(int)v)) * (1.0-(u-(int)u));
        float v1 = ((v-(int)v))*(1.0 - (u-(int)u));
        float v2 = (1.0 - (v-(int)v))*((u-(int)u));
        float v3 = ((v-(int)v))*((u-(int)u));
        pers[pers_tid] = static_cast<unsigned char>(
            pano[px0] * v0
            + pano[px1] * v1
            + pano[px2] * v2
            + pano[px3] * v3
        );
        pers[pers_tid + 1] = static_cast<unsigned char>(
            pano[px0 + 1] * v0
            + pano[px1 + 1] * v1
            + pano[px2 + 1] * v2
            + pano[px3 + 1] * v3
        );
        pers[pers_tid + 2] = static_cast<unsigned char>(
            pano[px0 + 2] * v0
            + pano[px1 + 2] * v1
            + pano[px2 + 2] * v2
            + pano[px3 + 2] * v3
        );
    }
}

void crop_gpu(
    const cv::Mat &pano,
    cv::Mat &pers,
    const cv::Mat &T_im2ori,
    int device)
{    
    // Calculate total number of bytes of input image and orientation matrix
    const int panoBytes = pano.step * pano.rows;
    const int persBytes = pers.step * pers.rows;

    // Return pointers
    unsigned char *d_pano, *d_pers;
    float *im2ori = new float[9];
    float *d_im2ori;

    // set device
    //FIXME: might want to add checks
    SAFE_CALL(cudaSetDevice(device), "CUDA Wrong device");
    
    // Allocate device memory
    //SAFE_CALL(cudaMallocManaged(&d_pano, panoBytes),"CUDA Malloc Failed");
    //SAFE_CALL(cudaMallocManaged(&d_pers, persBytes),"CUDA Malloc Failed");
    //SAFE_CALL(cudaMallocManaged(&d_im2ori, 9*sizeof(float)),"CUDA Malloc Failed");
    
    SAFE_CALL(cudaMalloc(&d_pano, panoBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_pers, persBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&d_im2ori, 9*sizeof(float)),"CUDA Malloc Failed");

    for(int i=0; i<3; i++)
    {
        im2ori[i*3] = T_im2ori.at<float>(i, 0);
        im2ori[i*3 + 1] = T_im2ori.at<float>(i, 1);
        im2ori[i*3 + 2] = T_im2ori.at<float>(i, 2);
    }

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_pano, pano.ptr(), panoBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_im2ori, im2ori, 9*sizeof(float), cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    // Specify device
    // int device = 0;
    // cudaGetDevice(&device);

    // Specify a reasonable block size 
    // (16, 16) is great most of the time
    // for better GPUs use (32, 32)
    const dim3 block(32, 32);
    //Calculate grid size to cover the whole image
    const dim3 grid((pers.cols + block.x - 1)/block.x, (pers.rows + block.y - 1)/block.y);
    
    
    // Launch the color conversion kernel
    // std::cout << pano.step << " " << pano.cols << " " << pano.rows << std::endl;
    crop_cuda<<<grid, block>>>(d_pano, d_pers, d_im2ori,
                                    pano.cols, pano.rows,
                                    pers.cols, pers.rows, 
                                    pano.step, pers.step);
    

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
    
    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(pers.ptr(), d_pers, persBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    
    // Free the device memory
    // Memory is not freed unless runtime is over or another process is run on the same GPU that uses the same thread
    SAFE_CALL(cudaFree(d_pano),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_pers),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_im2ori),"CUDA Free Failed");

    delete im2ori;
}
