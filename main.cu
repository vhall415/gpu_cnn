#include <stdio.h>
#include <stdint.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>

// function to check for errors
#define checkCUDNN(expression) 					\
{                                				\
    cudnnStatus_t status = (expression);                        \
    if(status != CUDNN_STATUS_SUCCESS) { 			\
	std::cerr << "Error on line " << __LINE__ << ": "	\
		  << cudnnGetErrorString(status) << std::endl;	\
	std::exit(EXIT_FAILURE);				\
    }	\
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

#define BATCH 1 // number of images
#define IN_CHANNELS 1   // number of channels of input image
#define FILTER_DIM 5    // side length of convolution filter size

int main(int argc, char* argv[]) {
    cv::Mat img = load_image("./gray_images/0.PNG");

    FILE *f;
    char buf[1000];
    // read weight files into arrays

    // conv1 weights
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
	
    // conv1 bias
    // 32
    float bias_conv1[32];
    f = fopen("./weights/var1.txt", "r");
    for(int i = 0; i < 32; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_conv1[i] = atof(buf);
    }
    fclose(f);

    // conv2 weigts
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

    //conv2 bias
    // 64
    f = fopen("./weights/var3.txt", "r");
    float bias_conv2[64];
    for(int i = 0; i < 64; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_conv2[i] = atof(buf);
    }
    fclose(f);
    
    // fully connected layer weights
    // 3136x1024
    f = fopen("./weights/var4_0.txt", "r");
    float *fully_con = new float[3136*1024];
    for(int i = 0; i < 1568*1024; i++) {
	    if(fgets(buf,1000,f) != NULL)
	        fully_con[i] = atof(buf);
    }
    fclose(f);

    f = fopen("./weights/var4_1.txt", "r");
    for(int i = 1568*1024; i < 3136*1024; i++) {
	    if(fgets(buf,1000,f) != NULL)
	        fully_con[i] = atof(buf);
    }
    fclose(f);

    float (*fc_mat)[1024] = new float[3136][1024];

    for(int row = 0; row < 3136; row++) {
	    for(int col = 0; col < 1024; col++) {
	        fc_mat[row][col] = fully_con[1024*row+col];
	    }
    }

    delete[] fully_con;

    // fully connected layer bias
    // 1024
    f = fopen("./weights/var5.txt", "r");
    float bias_fc[1024];
    for(int i = 0; i < 1024; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    bias_fc[i] = atof(buf);
    }
    fclose(f);

    // output layer weights
    // 1024x10
    f = fopen("./weights/var6.txt", "r");
    float out_mat[1024][10];
    for(int row = 0; row < 1024; row++) {
	for(int col = 0; col < 10; col++) {
	    if(fgets(buf,1000, f) != NULL)
	        out_mat[row][col] = atof(buf);
        }
    }
    fclose(f);

    // output layer bias
    // 10
    float bias_out[10];
    f = fopen("./weights/var7.txt", "r");
    for(int i = 0; i < 10; i++) {
        if(fgets(buf, 1000, f) != NULL) {
	        bias_out[i] = atof(buf);
	    }
    }
    fclose(f);

std::cerr << "1" << std::endl;

    // create handle for cudnn
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // conv 1 descriptors -------------------------------------------------------------------

    // create/set input tensor descriptor
    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/BATCH,
                                          /*channels=*/IN_CHANNELS,
                                          /*image_height=*/img.rows,
                                          /*image_width=*/img.cols));
    
    // create kernel descriptor
    cudnnFilterDescriptor_t conv1_kernel_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1_kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(conv1_kernel_desc,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/32,
                                          /*in_channels=*/IN_CHANNELS,
                                          /*kernel_height=*/FILTER_DIM,
                                          /*kernel_width=*/FILTER_DIM));

    // create convolution descriptor
    cudnnConvolutionDescriptor_t conv1_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv1_desc,
                                               /*pad_height=*/2,
                                               /*pad_width=*/2,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    // initialize variables for convolution 1 output dimensions
    int conv1_batch{0}, conv1_chan{0}, conv1_h{0}, conv1_w{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv1_desc,
                                                     in_desc,
                                                     conv1_kernel_desc,
                                                     &conv1_batch,
                                                     &conv1_chan,
                                                     &conv1_h,
                                                     &conv1_w));

    std::cerr << "Output Image: " << conv1_batch << " x " << conv1_h << " x " << conv1_w << " x " << conv1_chan << std::endl;

    // create output tensor descriptors
    cudnnTensorDescriptor_t conv1_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv1_out_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/conv1_batch,
                                          /*channels=*/conv1_chan,
                                          /*image_height=*/conv1_h,
                                          /*image_width=*/conv1_w));

    cudnnTensorDescriptor_t conv1_bias_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv1_bias_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/conv1_batch,
                                          /*channels=*/conv1_chan,
                                          /*image_height=*/1,
                                          /*image_width=*/1));


    // get forward convolution algorithm
    cudnnConvolutionFwdAlgo_t conv1_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   in_desc,
                                                   conv1_kernel_desc,
                                                   conv1_desc,
                                                   conv1_out_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   /*memoryLimitInBytes=*/0,
                                                   &conv1_alg));
    
    // get forward convolution workspace size
    size_t conv1_work{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       in_desc,
                                                       conv1_kernel_desc,
                                                       conv1_desc,
						                               conv1_out_desc,
                                                       conv1_alg,
                                                       &conv1_work));

    std::cerr << "Workspace size: " << (conv1_work / 1048576.0) << "MB" << std::endl;
    assert(conv1_work > 0);


    // relu descriptor -----------------------------------------------------------------------
    // same one used for all relu layers
    cudnnActivationDescriptor_t act_desc;
    checkCUDNN(cudnnCreateActivationDescriptor(&act_desc));
    checkCUDNN(cudnnSetActivationDescriptor(act_desc,
				                            CUDNN_ACTIVATION_RELU,
				                            CUDNN_PROPAGATE_NAN,
				                            /*relu_coef=*/0));


    // pool1 descriptors ---------------------------------------------------------------------
    // get variables for pooling 1 output dimensions

    cudnnPoolingDescriptor_t pool1_desc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pool1_desc));
    checkCUDNN(cudnnSetPooling2dDescriptor(pool1_desc,
				 /*mode=*/CUDNN_POOLING_MAX,
				 /*maxpoolingNanOpt=*/CUDNN_PROPAGATE_NAN,
				 /*windowHeight=*/2,
				 /*windowWidth=*/2,
				 /*verticalPadding=*/0,
				 /*horizontalPadding=*/0,
				 /*verticalStride=*/2,
				 /*horizontalStride=*/2));

    int pool1_batch{0}, pool1_chan{0}, pool1_h{0}, pool1_w{0};
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool1_desc,
                                      conv1_out_desc,
    				                  /*outN=*/&pool1_batch,
    			 	                  /*outC=*/&pool1_chan,
    				                  /*outH=*/&pool1_h,
    				                  /*outW=*/&pool1_w));

    std::cerr << "Pooling Output Size: " << pool1_batch << " x " << pool1_h << " x " << pool1_w << " x " << pool1_chan << std::endl;

    cudnnTensorDescriptor_t pool1_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&pool1_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(pool1_out_desc,
			       /*format=*/CUDNN_TENSOR_NHWC,
			       /*dataType=*/CUDNN_DATA_FLOAT,
			       /*batch_size=*/pool1_batch,
			       /*channels=*/pool1_chan,
		   	       /*image_height=*/pool1_h,
			       /*image_width=*/pool1_w));
    
    // conv2 descriptors ---------------------------------------------------------------------
    
    cudnnFilterDescriptor_t conv2_kernel_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&conv2_kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(conv2_kernel_desc,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/64,
                                          /*in_channels=*/pool1_chan,
                                          /*kernel_height=*/FILTER_DIM,
                                          /*kernel_width=*/FILTER_DIM));

    // create convolution descriptor
    cudnnConvolutionDescriptor_t conv2_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv2_desc,
                                               /*pad_height=*/2,
                                               /*pad_width=*/2,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    // initialize variables for convolution 2 output dimensions
    int conv2_batch{0}, conv2_chan{0}, conv2_h{0}, conv2_w{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv2_desc,
                                                     pool1_out_desc,
                                                     conv2_kernel_desc,
                                                     &conv2_batch,
                                                     &conv2_chan,
                                                     &conv2_h,
                                                     &conv2_w));

    std::cerr << "Conv2 Output Image: " << conv2_batch << " x " << conv2_h << " x " << conv2_w << " x " << conv2_chan << std::endl;

    // create output tensor descriptors
    cudnnTensorDescriptor_t conv2_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv2_out_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/conv2_batch,
                                          /*channels=*/conv2_chan,
                                          /*image_height=*/conv2_h,
                                          /*image_width=*/conv2_w));

    cudnnTensorDescriptor_t conv2_bias_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2_bias_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv2_bias_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/conv2_batch,
                                          /*channels=*/conv2_chan,
                                          /*image_height=*/1,
                                          /*image_width=*/1));
    // get forward convolution algorithm
    cudnnConvolutionFwdAlgo_t conv2_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   pool1_out_desc,
                                                   conv2_kernel_desc,
                                                   conv2_desc,
                                                   conv2_out_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   /*memoryLimitInBytes=*/0,
                                                   &conv2_alg));
    
    // get forward convolution workspace size
    size_t conv2_work{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       pool1_out_desc,
                                                       conv2_kernel_desc,
                                                       conv2_desc,
						                               conv2_out_desc,
                                                       conv2_alg,
                                                       &conv2_work));

    std::cerr << "Workspace size: " << (conv2_work / 1048576.0) << "MB" << std::endl;
    assert(conv2_work > 0);


    // pool2 descriptors ---------------------------------------------------------------------
    
    cudnnPoolingDescriptor_t pool2_desc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pool2_desc));
    checkCUDNN(cudnnSetPooling2dDescriptor(pool2_desc,
				 /*mode=*/CUDNN_POOLING_MAX,
				 /*maxpoolingNanOpt=*/CUDNN_PROPAGATE_NAN,
				 /*windowHeight=*/2,
				 /*windowWidth=*/2,
				 /*verticalPadding=*/0,
				 /*horizontalPadding=*/0,
				 /*verticalStride=*/2,
				 /*horizontalStride=*/2));

    int pool2_batch{0}, pool2_chan{0}, pool2_h{0}, pool2_w{0};
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool2_desc,
                                      conv2_out_desc,
    				                  /*outN=*/&pool2_batch,
    			 	                  /*outC=*/&pool2_chan,
    				                  /*outH=*/&pool2_h,
    				                  /*outW=*/&pool2_w));

    std::cerr << "Pool2 Output Size: " << pool2_batch << " x " << pool2_h << " x " << pool2_w << " x " << pool2_chan << std::endl;

    cudnnTensorDescriptor_t pool2_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&pool2_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(pool2_out_desc,
			       /*format=*/CUDNN_TENSOR_NHWC,
			       /*dataType=*/CUDNN_DATA_FLOAT,
			       /*batch_size=*/pool2_batch,
			       /*channels=*/pool2_chan,
		   	       /*image_height=*/pool2_h,
			       /*image_width=*/pool2_w));


    // fully connected layer descriptors -----------------------------------------------------
    
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;
    
    cublas_status = cublasCreate(&cublas_handle);

    cudnnTensorDescriptor_t fc_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&fc_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(fc_out_desc,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               pool2_batch,
                               1,
                               1,
                               1024));

    cudnnTensorDescriptor_t fc_bias_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&fc_bias_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(fc_bias_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/pool2_batch,
                                          /*channels=*/1,
                                          /*image_height=*/1,
                                          /*image_width=*/1024));

    // dropout layer descriptors -------------------------------------------------------------

    //void* states;

    cudnnDropoutDescriptor_t drop_desc;
    checkCUDNN(cudnnCreateDropoutDescriptor(&drop_desc));
    checkCUDNN(cudnnSetDropoutDescriptor(drop_desc,
                              cudnn,
                              /*dropout=*/0.1f,
                              /*states=*/NULL,
                              /*stateSizeInBytes=*/0,
                              /*seed=*/217));

    size_t drop_size{0};
    checkCUDNN(cudnnDropoutGetReserveSpaceSize(fc_out_desc,
                                    /*sizeInBytes=*/&drop_size));

    //size_t drop_state_size{0};
    //checkCUDNN(cudnnDropoutGetStatesSize(cudnn,
    //                                    &drop_state_size));

    cudnnTensorDescriptor_t drop_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&drop_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(drop_out_desc,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               pool2_batch,
                               1,
                               1,
                               1024));

    // output layer descriptors --------------------------------------------------------------

    cudnnTensorDescriptor_t out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               pool2_batch,
                               1,
                               1,
                               10));

    cudnnTensorDescriptor_t out_bias_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&out_bias_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(out_bias_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/pool2_batch,
                                          /*channels=*/1,
                                          /*image_height=*/1,
                                          /*image_width=*/10));
    // initialze device variables ---------------------------------------------
    int in_size = BATCH * IN_CHANNELS * img.rows * img.cols * sizeof(float);
    float* d_input{nullptr};
    cudaMalloc(&d_input, in_size);
    cudaMemcpy(d_input, img.ptr<float>(0), in_size, cudaMemcpyHostToDevice);
    
    float* d_kernel_conv1{nullptr};
    cudaMalloc(&d_kernel_conv1, sizeof(kernel_conv1));
    cudaMemcpy(d_kernel_conv1, kernel_conv1, sizeof(kernel_conv1), cudaMemcpyHostToDevice);
    
    void* d_conv1_work{nullptr};
    cudaMalloc(&d_conv1_work, conv1_work);
    
    int conv1_size = conv1_batch * conv1_chan * conv1_h * conv1_w * sizeof(float);
    float* d_conv1_out{nullptr};
    cudaMalloc(&d_conv1_out, conv1_size);
    cudaMemset(d_conv1_out, 0, conv1_size);
    
    float* d_bias_conv1{nullptr};
    cudaMalloc(&d_bias_conv1, sizeof(bias_conv1));
    cudaMemcpy(d_bias_conv1, bias_conv1, sizeof(bias_conv1), cudaMemcpyHostToDevice);
    
    int pool1_size = pool1_batch * pool1_chan * pool1_h * pool1_w * sizeof(float);
    float* d_pool1_out{nullptr};
    cudaMalloc(&d_pool1_out, pool1_size);
    cudaMemset(d_pool1_out, 0, pool1_size);

    float* d_kernel_conv2{nullptr};
    cudaMalloc(&d_kernel_conv2, sizeof(kernel_conv2));
    cudaMemcpy(d_kernel_conv2, kernel_conv2, sizeof(kernel_conv2), cudaMemcpyHostToDevice);
    
    void* d_conv2_work{nullptr};
    cudaMalloc(&d_conv2_work, conv2_work);
    
    int conv2_size = conv2_batch * conv2_chan * conv2_h * conv2_w * sizeof(float);
    float* d_conv2_out{nullptr};
    cudaMalloc(&d_conv2_out, conv2_size);
    cudaMemset(d_conv2_out, 0, conv2_size);

    float* d_bias_conv2{nullptr};
    cudaMalloc(&d_bias_conv2, sizeof(bias_conv2));
    cudaMemcpy(d_bias_conv2, bias_conv2, sizeof(bias_conv2), cudaMemcpyHostToDevice);
    
    int pool2_size = pool2_batch * pool2_chan * pool2_h * pool2_w * sizeof(float);
    float* d_pool2_out{nullptr};
    cudaMalloc(&d_pool2_out, pool2_size);
    cudaMemset(d_pool2_out, 0, pool2_size);

    int fc_size = 3136*1024*sizeof(float);
    float* d_fully_con_mat{nullptr};
    cudaMalloc(&d_fully_con_mat, fc_size);
    cublas_status = cublasSetMatrix(3136, 1024, fc_size, fully_con, 3136, d_fully_con_mat, 3136);

    float* d_bias_fc{nullptr};
    cudaMalloc(&d_bias_fc, sizeof(bias_fc));
    cudaMemcpy(d_bias_fc, bias_fc, sizeof(bias_fc), cudaMemcpyHostToDevice);
    
    int fc_out_size = 1024 * sizeof(float);
    float* d_fully_con_out{nullptr};
    cudaMalloc(&d_fully_con_out, fc_out_size);
    cudaMemset(d_fully_con_out, 0, fc_out_size);

    void* d_reserve{nullptr};
    cudaMalloc(&d_reserve, drop_size);

    float* d_drop_out{nullptr};
    cudaMalloc(&d_drop_out, fc_size);
    cudaMemset(d_drop_out, 0, fc_size);

    int num_elems = pool2_chan * 1024 * 10;
    float* d_out_mat{nullptr};
    cudaMalloc(&d_out_mat, num_elems);
    cublas_status = cublasSetMatrix(1024, 10, num_elems*sizeof(float), out_mat, 1024, d_out_mat, 1024);

    float* d_bias_out{nullptr};
    cudaMalloc(&d_bias_out, sizeof(bias_out));
    cudaMemcpy(d_bias_out, bias_out, sizeof(bias_out), cudaMemcpyHostToDevice);
    
    int out_size = 10 * sizeof(float);
    float* d_out{nullptr};
    cudaMalloc(&d_out, out_size);
    cudaMemset(d_out, 0, out_size);

    const float alpha = 1.0f, beta = 0.0f;

    // convolution 1 layer ----------------------------------------------
    // map grayscale input to 32 feature maps
    // 28x28x1 -> 28x28x32

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       in_desc,
                                       d_input,
                                       conv1_kernel_desc,
                                       d_kernel_conv1,
                                       conv1_desc,
                                       conv1_alg,
                                       d_conv1_work,
                                       conv1_work,
                                       &beta,
                                       conv1_out_desc,
                                       d_conv1_out));

    checkCUDNN(cudnnAddTensor(cudnn,
			      &alpha,
			      conv1_bias_desc,
   			      d_bias_conv1,
			      &alpha,
			      conv1_out_desc,
			      d_conv1_out));

    // relu 1 layer (activation) -----------------------------------------

    checkCUDNN(cudnnActivationForward(cudnn,
			   act_desc,
			   &alpha,
			   conv1_out_desc,
			   d_conv1_out,
			   &beta,
			   conv1_out_desc,
			   d_conv1_out));

    // pooling 1 layer -------------------------------------------------
    // downsample by 2x
    // 28x28x32 -> 14x14x32
    
    checkCUDNN(cudnnPoolingForward(cudnn,
			pool1_desc,
			&alpha,
			conv1_out_desc,
			d_conv1_out,
			&beta,
			pool1_out_desc,
			d_pool1_out));

    // convolution 2 layer -------------------------------------------
    // map 32 feature maps to 64
    // 2x2 padding
    // 14x14x32 -> 14x14x64

    checkCUDNN(cudnnConvolutionForward(cudnn,
                            &alpha,
                            pool1_out_desc,
                            d_pool1_out,
                            conv2_kernel_desc,
                            d_kernel_conv2,
                            conv2_desc,
                            conv2_alg,
                            d_conv2_work,
                            conv2_work,
                            &beta,
                            conv2_out_desc,
                            d_conv2_out));

    checkCUDNN(cudnnAddTensor(cudnn,
			      &alpha,
			      conv2_bias_desc,
   			      d_bias_conv2,
			      &alpha,
			      conv2_out_desc,
			      d_conv2_out));

    // relu 2 layer ---------------------------------------------------

    checkCUDNN(cudnnActivationForward(cudnn,
                           act_desc,
                           &alpha,
                           conv2_out_desc,
                           d_conv2_out,
                           &beta,
                           conv2_out_desc,
                           d_conv2_out));

    // pooling 2 layer -----------------------------------------------
    // downsample by 2x
    // 14x14x64 -> 7x7x64

    checkCUDNN(cudnnPoolingForward(cudnn,
                        pool2_desc,
                        &alpha,
                        conv2_out_desc,
                        d_conv2_out,
                        &beta,
                        pool2_out_desc,
                        d_pool2_out));

    // fully connected 1 layer ---------------------------------------
    // map 7x7x64 -> 1024 features

    cublas_status = cublasSgemm(cublas_handle,
                                /*transa=*/CUBLAS_OP_N,
                                /*transb=*/CUBLAS_OP_N,
                                /*m=*/1,
                                /*n=*/7*7*64,
                                /*k=*/1024,
                                /*alpha=*/&alpha,
                                /*A=*/d_fully_con_mat,
                                /*lda=*/1,
                                /*B=*/d_pool2_out,
                                /*ldb=*/7*7*64,
                                /*beta=*/&beta,
                                /*C=*/d_fully_con_out,
                                /*ldc=*/1);

    checkCUDNN(cudnnAddTensor(cudnn,
			      &alpha,
			      fc_bias_desc,
   			      d_bias_fc,
			      &alpha,
			      fc_out_desc,
			      d_fully_con_out));


    // relu 3 layer -------------------------------------------------


    checkCUDNN(cudnnActivationForward(cudnn,
                           act_desc,
                           &alpha,
                           fc_out_desc,
                           d_fully_con_out,
                           &beta,
                           fc_out_desc,
                           d_fully_con_out));

    // dropout layer -----------------------------------------------


 
    // output layer --------------------------------------------------------------------------
    // map 1024 features to 10 classes (one for each digit)

    cublas_status = cublasSgemm(cublas_handle,
                                /*transa=*/CUBLAS_OP_N,
                                /*transb=*/CUBLAS_OP_N,
                                /*m=*/1,
                                /*n=*/1024,
                                /*k=*/10,
                                /*alpha=*/&alpha,
                                /*A=*/d_fully_con_out,
                                /*lda=*/1,
                                /*B=*/d_out_mat,
                                /*ldb=*/1024,
                                /*beta=*/&beta,
                                /*C=*/d_out,
                                /*ldc=*/1);
    // add bias to d_out
    checkCUDNN(cudnnAddTensor(cudnn,
			      &alpha,
			      out_bias_desc,
   			      d_bias_out,
			      &alpha,
			      out_desc,
			      d_out));


    //float* h_full_out = new float[fc_out_size];
    //cudaMemcpy(h_full_out, d_fully_con_out, fc_out_size, cudaMemcpyDeviceToHost);

    //std::cerr << h_full_out[0] << std::endl;

    float* h_out = new float[conv1_size];
    cudaMemcpy(h_out, d_conv1_out, conv1_size, cudaMemcpyDeviceToHost);
    
    save_image("./out.png", h_out, conv1_h, conv1_w);
    //delete[] h_full_out;
    delete[] h_out;
    delete[] fc_mat;

    cudaFree(d_input);
    cudaFree(d_kernel_conv1);
    cudaFree(d_conv1_work);
    cudaFree(d_conv1_out);
    cudaFree(d_pool1_out);
    cudaFree(d_kernel_conv2);
    cudaFree(d_conv2_work);
    cudaFree(d_conv2_out);
    cudaFree(d_pool2_out);
    cudaFree(d_fully_con_mat);
    cudaFree(d_fully_con_out);
    cudaFree(d_reserve);
    cudaFree(d_drop_out);
    cudaFree(d_out_mat);
    cudaFree(d_out);
    cudaFree(d_bias_conv1);
    cudaFree(d_bias_conv2);
    cudaFree(d_bias_fc);
    cudaFree(d_bias_out);

    cublasDestroy(cublas_handle);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyFilterDescriptor(conv1_kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyTensorDescriptor(conv1_out_desc);
    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroyPoolingDescriptor(pool1_desc);
    cudnnDestroyTensorDescriptor(pool1_out_desc);
    cudnnDestroyFilterDescriptor(conv2_kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv2_desc);
    cudnnDestroyTensorDescriptor(conv2_out_desc);
    cudnnDestroyPoolingDescriptor(pool2_desc);
    cudnnDestroyTensorDescriptor(pool2_out_desc);
    cudnnDestroyTensorDescriptor(fc_out_desc);
    cudnnDestroyDropoutDescriptor(drop_desc);
    cudnnDestroyTensorDescriptor(drop_out_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);
    cudnnDestroyTensorDescriptor(fc_bias_desc);
    cudnnDestroyTensorDescriptor(out_bias_desc);

    
    cudnnDestroy(cudnn);
}
