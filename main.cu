#include <stdio.h>
#include <stdint.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#include "kernel.cu"

// function to check for errors
#define checkCUDNN(expression) {                                \
    cudnnStatus_t status = (expression);                        \
    if(status != CUDNN_STATUS_SUCCESS) {                        \
        std::cerr << "Error on line " << __LINE__ < ": "        \
                  << cudnnGetErrorString(status) << std::endl;  \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}                                                               \

// use opencv to load/save an image from a path
cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

void save_image(const char* output_filename, float* buffer, int height, int width) {
    cv::Mat output_image(height, width, CV32FC3, buffer);
    //Make negative values zero
    cv::threshold(output_image, output_image, /*threshold=*/0, /*maxval=*/0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV8UC3);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}

int main(int argc, char* argv[]) {
    Timer timer;

    // initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    // initialize device variables --------------------------------------------
    
    // create handle for cudnn
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // create/set input tensor descriptor
    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/1,
                                          /*image_height=*/image.rows,
                                          /*image_width=*/image.cols));
    
    // create kernel descriptor
    cudnnFilterDescriptor_t kernel_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_desc,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/1,
                                          /*in_channels=*/1,
                                          /*kernel_height=*/3,  //5
                                          /*kernel_width=*/3)); //5

    // create convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                               /*pad_height=*/2,
                                               /*pad_width=*/2,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CONVOLUTION, //CUDNN_CROSS_CORRELATION
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    // initialize variables for convolution
    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                                     in_desc,
                                                     kernel_desc,
                                                     &bathc_size,
                                                     &channels,
                                                     &height,
                                                     &width));

    std::cerr << "Output Image: " << height << " x " << width << " x " << channels << std::endl;

    // create output tensor descriptor
    cudnnTensorDescriptor_t out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc,
                                           /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*batch_size=*/1,
                                           /*channels=*/1,
                                           /*image_height=*/image.rows,
                                           /*image_width=*/image.cols));

    // forward convolution ----------------------------------------------------
    cudnnConvolutionFwdAlgo_t conv_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   in_desc,
                                                   kernel_desc,
                                                   conv_desc,
                                                   out_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   /*memoryLimitInBytes=*/0,
                                                   &conv_alg));
    
    // get workspace size
    size_t workspace_bytes{0}
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       in_desc,
                                                       kernel_desc,
                                                       conv_desc,
                                                       conv_alg,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    assert(workspace_bytes > 0);

    // initialze device variables ---------------------------------------------
    void d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float* d_input {nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
    
    float* d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    const float kernel_temp[5][5] = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}
    };

    float h_kernel[3][1][3][3];
    for(int kernel = 0; kernel < 3; ++kernel) {
        for(int channel = 0; channel < 1; ++channel) {
            for(int row = 0; row < 3; ++row) {
                for(int col = 0; col < 3; ++col) {
                    h_kernel[kernel][channel][row][col] = kernel_temp[row][col];
                }
            }
        }
    }

    flowt* d_kernel{nullptr}
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       in_desc,
                                       d_input,
                                       kernel_desc,
                                       d_kernel,
                                       conv_desc,
                                       conv_alg,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       out_desc,
                                       d_output));

    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    save_image("cudnn_out.png", h_output, height, width);

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    cudnnDestroy(cudnn);
}
