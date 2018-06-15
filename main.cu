#include <stdio.h>
#include <stdint.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>

// function to check for errors
#define checkCUDNN(expression) {				\
    cudnnStatus_t status = (expression);			\
    if(status != CUDNN_STATUS_SUCCESS) {				\
	std::cerr << "Error on line " << __LINE__ << ": "	\
		  << cudnnGetErrorString(status) << std::endl;	\
	std::exit(EXIT_FAILURE);				\
    }								\
}

// use opencv to lead/save an image

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32FC1);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    std::cerr << "Input image: " << image.rows << " x " << image.cols << " x "
	      << image.channels() << std::endl;
    return image;
}

void save_image(const char* output_filename, float* buffer, int height, int width) {
    cv::Mat output_image(height, width, CV_32FC1, buffer);
    // Make negative values zero
    cv::threshold(output_image, output_image, /*thershold=*/0, /*maxval=*/0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC1);
    cv::imwrite(output_filename, output_image);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}

#define BATCH 1
#define IN_CHANNELS 1

int main(int argc, char* argv[]) {
    cv::Mat img = load_image("./gray_images/0.PNG");

    std::cout << img << std::endl;

    for(int i = 0; i < img.rows; i++) {
	for(int j = 0; j < img.cols; j++) {
	    //std::cout << img.at(i,j) << "\t";
	}
	std::cout << std::endl;
    }

    FILE *f;
    char buf[1000];
    // read weight files into arrays

    // conv1 weights
    // 5x5x1x32
    f = fopen("./weights/var0.txt", "r");
    float h_conv1_kernel[32][1][5][5];
    for(int kernel = 0; kernel < 32; kernel++) {
	for(int channel = 0; channel < 1; channel++) {
	    for(int row = 0; row < 5; row++) {
		for(int col = 0; col < 5; col++) {
		    if(fgets(buf,1000,f) != NULL)
		    	h_conv1_kernel[kernel][channel][row][col] = atof(buf);
		}
	    }
	}
    }
    fclose(f);

    for(int k = 0; k < 32; k++) {
	std::cout << "Kernel " << k << ":" << std::endl;
	for(int ch = 0; ch < 1; ch++) {
	    for(int r = 0; r < 5; r++) {
		for(int c = 0; c < 5; c++) {
		    std::cout << h_conv1_kernel[k][ch][r][c] << " ";
		}
		std::cout << std::endl;
	    }
	    std::cout << std::endl;
	}
    }
	
    // conv1 bias
    // 32
    float h_conv1_bias[32];
    f = fopen("./weights/var1.txt", "r");
    for(int i = 0; i < 32; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    h_conv1_bias[i] = atof(buf);
    }
    fclose(f);

    // conv2 weigts
    // 5x5x32x64
    f = fopen("./weights/var2.txt", "r");
    float h_conv2_kernel[64][32][5][5];
    for(int kernel = 0; kernel < 64; kernel++) {
	for(int channel = 0; channel < 32; channel++) {
	    for(int row = 0; row < 5; row++) {
		for(int col = 0; col < 5; col++) {
		    if(fgets(buf, 1000, f) != NULL)
			h_conv2_kernel[kernel][channel][row][col] = atof(buf);
		}
	    }
	}
    }
    fclose(f);

    //conv2 bias
    // 64
    f = fopen("./weights/var3.txt", "r");
    float h_conv2_bias[64];
    for(int i = 0; i < 64; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    h_conv2_bias[i] = atof(buf);
    }
    fclose(f);

    // fully connected layer weights
    // 3136x1024
    f = fopen("./weights/var4.txt", "r");
    float h_fc_weight[3136][1024];
    for(int row = 0; row < 3136; row++) {
	for(int col = 0; col < 1024; col++) {
 	    if(fgets(buf, 1000, f) != NULL)
	        h_fc_weight[row][col] = atof(buf);
	}
    }
    fclose(f);

    // fully connected layer bias
    // 1024
    f = fopen("./weights/var5.txt", "r");
    float h_fc_bias[1024];
    for(int i = 0; i < 1024; i++) {
	if(fgets(buf, 1000, f) != NULL)
	    h_fc_bias[i] = atof(buf);
    }
    fclose(f);

    // output layer weights
    // 1024x10
    f = fopen("./weights/var6.txt", "r");
    float h_out_weight[1024][10];
    for(int row = 0; row < 1024; row++) {
	for(int col = 0; col < 10; col++) {
	    if(fgets(buf,1000, f) != NULL)
	        h_out_weight[row][col] = atof(buf);
        }
    }
    fclose(f);

    // output layer bias
    // 10
    float h_out_bias[10];
    f = fopen("./weights/var7.txt", "r");
    for(int i = 0; i < 10; i++) {
        if(fgets(buf, 1000, f) != NULL) {
	    h_out_bias[i] = atof(buf);
	}
    }
    fclose(f);

    // create handle for cudnn
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // conv1 variables --------------------------------------------------------

    // input tensor
    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
					  /*format=*/CUDNN_TENSOR_NHWC,
					  /*dataType=*/CUDNN_DATA_FLOAT,
					  /*batch_size=*/1,
					  /*channels=*/1,
					  /*image_height=*/img.rows,
					  /*image_width=*/img.cols));

    // kernel descriptor
    cudnnFilterDescriptor_t conv1_kernel_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1_kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(conv1_kernel_desc,
					  /*dataType=*/CUDNN_DATA_FLOAT,
					  /*format=*/CUDNN_TENSOR_NCHW,
					  /*out_channels=*/1,
					  /*in_channels=*/1,
					  /*kernel_height=*/5,
					  /*kernel_width=*/5));

    // convolution descriptor
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

    // conv1 output dimensions
    int conv1_batch{0}, conv1_chan{0}, conv1_h{0}, conv1_w{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv1_desc,
						     in_desc,
						     conv1_kernel_desc,
						     &conv1_batch,
						     &conv1_chan,
						     &conv1_h,
						     &conv1_w));

    std::cerr << "Output image: " << conv1_h << " x " << conv1_w << " x "
	      << conv1_chan << std::endl;

    // conv1 output tensor
    cudnnTensorDescriptor_t conv1_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv1_out_desc,
					  /*format=*/CUDNN_TENSOR_NHWC,
					  /*dataType=*/CUDNN_DATA_FLOAT,
					  /*batch_size=*/conv1_batch,
					  /*channels=*/conv1_chan,
					  /*image_height=*/conv1_h,
					  /*image_width=*/conv1_w));

    // conv1 forward algorithm
    cudnnConvolutionFwdAlgo_t conv1_alg;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
						   in_desc,
						   conv1_kernel_desc,
						   conv1_desc,
						   conv1_out_desc,
						   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
						   /*memoryLimitInBytes=*/0,
						   &conv1_alg));

    // conv1 workspace size
    size_t conv1_workspace{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
						       in_desc,
						       conv1_kernel_desc,
						       conv1_desc,
						       conv1_out_desc,
						       conv1_alg,
						       &conv1_workspace));

    std::cerr << "Workspace size: " << (conv1_workspace / 1048576.0)
	      << "MB" << std::endl;
    assert(conv1_workspace > 0);

    // initialize device variables --------------------------------------------

    int in_size = BATCH * IN_CHANNELS * img.rows * img.cols * sizeof(float);
    float* d_in{nullptr};
    cudaMalloc(&d_in, in_size);
    cudaMemcpy(d_in, img.ptr<float>(0), in_size, cudaMemcpyHostToDevice);

    float* d_conv1_kernel{nullptr};
    cudaMalloc(&d_conv1_kernel, sizeof(h_conv1_kernel));
    cudaMemcpy(d_conv1_kernel, h_conv1_kernel, sizeof(h_conv1_kernel), cudaMemcpyHostToDevice);

    void* d_conv1_work{nullptr};
    cudaMalloc(&d_conv1_work, conv1_workspace);

    int conv1_out_size = conv1_batch * conv1_chan * conv1_h * conv1_w * sizeof(float);
    float* d_conv1_out{nullptr};
    cudaMalloc(&d_conv1_out, conv1_out_size);
    cudaMemset(d_conv1_out, 0, conv1_out_size);

    const float alpha = 1.0f, beta = 0.0f;

    // conv1 ------------------------------------------------------------------

    checkCUDNN(cudnnConvolutionForward(cudnn,
				       &alpha,
				       in_desc,
				       d_in,
				       conv1_kernel_desc,
				       d_conv1_kernel,
				       conv1_desc,
				       conv1_alg,
				       d_conv1_work,
				       conv1_workspace,
				       &beta,
				       conv1_out_desc,
				       d_conv1_out));

    // relu1 ------------------------------------------------------------------



    // pool1 ------------------------------------------------------------------



    // conv2 ------------------------------------------------------------------



    // relu2 ------------------------------------------------------------------



    // pool2 ------------------------------------------------------------------



    // fc ---------------------------------------------------------------------



    // relu3 ------------------------------------------------------------------



    // dropout ----------------------------------------------------------------



    // output -----------------------------------------------------------------



    // copy output to host side

    float* h_out = new float[conv1_out_size];
    cudaMemcpy(h_out, d_conv1_out, conv1_out_size, cudaMemcpyDeviceToHost);

    std::cout << "Output Data:" << std::endl;
    //for(int i = 0; i < 10; i++) {
    //    std::cout << h_out[i] << std::endl;
    //}

    //save_image("./out_test.png", h_out, conv1_h, conv1_w);

    // free variables

    delete[] h_out;

    cudaFree(d_in);
    cudaFree(d_conv1_kernel);
    cudaFree(d_conv1_work);
    cudaFree(d_conv1_out);

    cudnnDestroy(cudnn);
    
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyFilterDescriptor(conv1_kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyTensorDescriptor(conv1_out_desc);

    return 0;
}
