#include <stdio.h>
#include <stdint.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>

// function to check for errors
#define checkCUDNN(expression) \
{                                \
    cudnnStatus_t status = (expression);                        \
    if(status != CUDNN_STATUS_SUCCESS) {                        \
        std::cerr << "Error on line " << __LINE__ < ": " << cudnnGetErrorString(status) << std::endl;  \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}

// use opencv to load/save an image from a path
cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    std::cerr << "Input image: " << image.rows << " x " << image.cols << " x " << image.channels() << std::endl;
    return image;
}

void save_image(const char* output_filename, float* buffer, int height, int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer);
    //Make negative values zero
    cv::threshold(output_image, output_image, /*threshold=*/0, /*maxval=*/0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3);
    cv::imwrite(output_filename, output_image);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}

int main(int argc, char* argv[]) {
    cv::Mat img = load_image("./bw_images/0.PNG");

    FILE *f;
    char buf[1000];
    // read weight files into arrays
    // 5x5x1x32
    f = fopen("./weights/var0.txt", "r");
    float kernel_conv1[32][1][5][5];
    for(int kernel = 0; kernel < 32; kernel++) {
	for(int channel = 0; channel < 1; channel++) {
	    for(int row = 0; row < 5; row++) {
		for(int col = 0; col < 5; col++) {
		    if(fgets(buf,1000,f) != NULL)
		    	kernel_conv1[kernel][channel][row][col] = atof(buf);
		}
	    }
	}
    }
    fclose(f);
	
    // 32
    float bias_conv1[32];
    f = fopen("./weights/var1.txt", "r");
    for(int i = 0; i < 32; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_conv1[i] = atof(buf);
    }
    fclose(f);

    // 5x5x32x64
    f = fopen("./weights/var2.txt", "r");
    float kernel_conv2[64][32][5][5];
    for(int kernel = 0; kernel < 64; kernel++) {
	for(int channel = 0; channel < 32; channel++) {
	    for(int row = 0; row < 5; row++) {
		for(int col = 0; col < 5; col++) {
		    if(fgets(buf, 1000, f) != NULL)
			kernel_conv2[kernel][channel][row][col] = atof(buf);
		}
	    }
	}
    }
    fclose(f);

    // 64
    f = fopen("./weights/var3.txt", "r");
    float bias_conv2[64];
    for(int i = 0; i < 64; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_conv2[i] = atof(buf);
    }
    fclose(f);

    // 3136x1024
    f = fopen("./weights/var4.txt", "r");
    float fully_con[3136][1024];
    for(int row = 0; row < 3136; row++) {
	for(int col = 0; col < 1024; col++) {
 	    if(fgets(buf, 1000, f) != NULL)
	        fully_con[row][col] = atof(buf);
	}
    }
    fclose(f);

    // 1024
    f = fopen("./weights/var5.txt", "r");
    float bias_fully_con[1024];
    for(int i = 0; i < 1024; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_fully_con[i] = atof(buf);
    }
    fclose(f);

    // 1024x10
    f = fopen("./weights/var6.txt", "r");
    float drop[1024][10];
    for(int row = 0; row < 1024; row++) {
	for(int col = 0; col < 10; col++) {
	    if(fgets(buf,1000, f) != NULL)
	        drop[row][col] = atof(buf);
        }
    }
    fclose(f);

    // 10
    float softmax[10];
    f = fopen("./weights/var7.txt", "r");
    for(int i = 0; i < 10; i++) {
        if(fgets(buf, 1000, f) != NULL) {
	    softmax[i] = atof(buf);
	}
    }
    fclose(f);

    // create handle for cudnn
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // start of first convolutional layer -------------------------------------

    // create/set input tensor descriptor
    cudnnTensorDescriptor_t in_desc;
    /*checkCUDNN(*/cudnnCreateTensorDescriptor(&in_desc);
    /*checkCUDNN(*/cudnnSetTensor4dDescriptor(in_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/1,
                                          /*image_height=*/img.rows,
                                          /*image_width=*/img.cols);
    
    // create kernel descriptor
    cudnnFilterDescriptor_t conv1_kernel_desc;
    /*checkCUDNN(*/cudnnCreateFilterDescriptor(&conv1_kernel_desc);
    /*checkCUDNN(*/cudnnSetFilter4dDescriptor(conv1_kernel_desc,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/32,
                                          /*in_channels=*/1,
                                          /*kernel_height=*/5,
                                          /*kernel_width=*/5);

    // create convolution descriptor
    cudnnConvolutionDescriptor_t conv1_desc;
    /*checkCUDNN(*/cudnnCreateConvolutionDescriptor(&conv1_desc);
    /*checkCUDNN(*/cudnnSetConvolution2dDescriptor(conv1_desc,
                                               /*pad_height=*/2,
                                               /*pad_width=*/2,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT);

    // initialize variables for convolution 1 output dimensions
    int batch_size{0}, channels{0}, height{0}, width{0};
    /*checkCUDNN(*/cudnnGetConvolution2dForwardOutputDim(conv1_desc,
                                                     in_desc,
                                                     conv1_kernel_desc,
                                                     &batch_size,
                                                     &channels,
                                                     &height,
                                                     &width);

    std::cerr << "Output Image: " << conv1_batch << " x " << conv1_h << " x " << width << " x " << channels << std::endl;

    // create output tensor descriptors
    cudnnTensorDescriptor_t conv1_out_desc;
    /*checkCUDNN(*/cudnnCreateTensorDescriptor(&conv1_out_desc);
    /*checkCUDNN(*/cudnnSetTensor4dDescriptor(conv1_out_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/channels,
                                          /*image_height=*/height,
                                          /*image_width=*/width);

    // get variables for pooling 1 output dimensions
 
    cudnnPoolingDescriptor_t pool1_desc;
    cudnnCreatePoolingDescriptor(&pool1_desc);
    cudnnSetActivationDescriptor(pool1_desc,
				 /*mode=*/CUDNN_POOLING_MAX,
				 /*maxpoolingNanOpt=*/CUDNN_PROPAGATE_NAN,
				 /*windowHeight=*/2,
				 /*windowWidth=*/2,
				 /*verticalPadding=*/1,
				 /*horizontalPadding=*/1,
				 /*verticalStride=*/2,
				 /*horizontalStride=*/2);

    int pool1_batch{0}, pool1_chan{0}, pool1_h{0}, pool1_w{0};
    cudnnGetPooling2dForwardOutputDim(pool_desc,
                                      conv1_out_desc,
    				      /*outN=*/&pool1_batch,
    			 	      /*outC=*/&pool1_chan,
    				      /*outH=*/&pool1_h,
    				      /*outW=*/&pool1_w)

    std::cerr << "Pooling Output Size: " << pool1_batch << " x " << pool1_h << " x " << pool1_w << " x " << pool1_chan << std::endl;

    cudnnTensorDescriptor_t pool1_out_desc;
    cudnnCreateTensorDescriptor(&pool1_out_desc);
    cudnnSetTensor4dDescriptor(pool1_out_desc,
			       /*format=*/CUDNN_TENSOR_NHWC,
			       /*dataType=*/CUDNN_DATA_FLOAT,
			       /*batch_size=*/pool1_batch,
			       /*channels=*/pool1_chan,
		   	       /*image_height=*/pool1_h,
			       /*image_width=*/pool1_w);
    
    cudnnTensorDescriptor_t conv2_out_desc;
    /*checkCUDNN(*/cudnnCreateTensorDescriptor(&conv2_out_desc);
    /*checkCUDNN(*/cudnnSetTensor4dDescriptor(conv2_out_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/64,
                                          /*image_height=*/img.rows/2,
                                          /*image_width=*/img.cols/2);

    cudnnTensorDescriptor_t pool2_out_desc;
    cudnnCreateTensorDescriptor(&pool2_out_desc);
    cudnnSetTensor4dDescriptor(pool2_out_desc,
			       /*format=*/CUDNN_TENSOR_NHWC,
			       /*dataType=*/CUDNN_DATA_FLOAT,
			       /*batch_size=*/1,
			       /*channels=*/64,
		   	       /*image_height=*/img.rows/4,
			       /*image_width=*/img.cols/4);
 

    // get forward convolution algorithm
    cudnnConvolutionFwdAlgo_t conv_alg;
    /*checkCUDNN(*/cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   in_desc,
                                                   kernel_desc,
                                                   conv_desc,
                                                   conv1_out_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   /*memoryLimitInBytes=*/0,
                                                   &conv_alg);
    
    // get forward convolution workspace size
    size_t workspace_bytes{0};
    /*checkCUDNN(*/cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       in_desc,
                                                       kernel_desc,
                                                       conv_desc,
						       conv1_out_desc,
                                                       conv_alg,
                                                       &workspace_bytes);
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    assert(workspace_bytes > 0);

    // initialze device variables ---------------------------------------------
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);
    std::cout << "image_bytes: " << image_bytes << std::endl;

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, img.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
    
    float* d_conv1_out{nullptr};
    cudaMalloc(&d_conv1_out, image_bytes);
    cudaMemset(d_conv1_out, 0, image_bytes);

    float* d_kernel_conv1{nullptr};
    cudaMalloc(&d_kernel_conv1, sizeof(kernel_conv1));
    cudaMemcpy(d_kernel_conv1, kernel_conv1, sizeof(kernel_conv1), cudaMemcpyHostToDevice);

    float* d_kernel_conv2{nullptr};
    cudaMalloc(&d_kernel_conv2, sizeof(kernel_conv2));
    cudaMemcpy(d_kernel_conv2, kernel_conv2, sizeof(kernel_conv2), cudaMemcpyHostToDevice);
    
    int pool1_size = pool1_batch * pool1_chan * pool1_h * pool1_w * sizeof(float);
    float* d_pool1_out{nullptr};
    cudaMalloc(&d_pool1_out, pool1_size);
    cudaMemset(d_pool1_out, 0, pool1_size);


    const float alpha = 1.0f, beta = 0.0f;

    // convolution 1 layer ----------------------------------------------
    // map grayscale input to 32 feature maps
    // 28x28x1 -> 28x28x32

    /*checkCUDNN(*/cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       in_desc,
                                       d_input,
                                       kernel_desc,
                                       d_kernel_conv1,
                                       conv_desc,
                                       conv_alg,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       conv1_out_desc,
                                       d_conv1_out);

    // relu 1 layer (activation) -----------------------------------------

    cudnnActivationDescriptor_t act_desc;
    cudnnCreateActivationDescriptor(&act_desc);
    cudnnSetActivationDescriptor(act_desc,
				 CUDNN_ACTIVATION_RELU,
				 CUDNN_PROPAGATE_NAN,
				 /*relu_coef=*/0);

    cudnnActivationForward(cudnn,
			   act_desc,
			   &alpha,
			   conv1_out_desc,
			   &d_conv1_out,
			   &beta,
			   conv1_out_desc,
			   d_conv1_out);

    // pooling 1 layer -------------------------------------------------
    // downsample by 2x
    // 28x28x32 -> 14x14x32
    

    cudnnPoolingForward(cudnn,
			pool_desc,
			&alpha,
			conv1_out_desc,
			d_conv1_out,
			&beta,
			pool1_out_desc,
			d_pool1_out);

    // convolution 2 layer -------------------------------------------
    // map 32 feature maps to 64
    // 2x2 padding
    // 14x14x32 -> 14x14x64



    // relu 2 layer ---------------------------------------------------



    // pooling 2 layer -----------------------------------------------
    // downsample by 2x
    // 14x14x64 -> 7x7x64



    // fully connected 1 layer ---------------------------------------
    // map 7x7x64 -> 1024 features



    // relu 3 layer -------------------------------------------------



    // dropout layer -----------------------------------------------
    // control complexity of model



    // softmax layer -----------------------------------------------
    // map 1024 features to 10 classes (one for each digit)



    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
   
    save_image("./conv1_out.png", h_output, height, width);

    delete[] h_output;
    cudaFree(d_kernel_conv1);
    cudaFree(d_kernel_conv2);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    cudnnDestroy(cudnn);
}
