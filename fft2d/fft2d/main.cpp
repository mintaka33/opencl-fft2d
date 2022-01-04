#include <stdio.h>
#include <vector>
#include <memory>
#include <string>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include <fstream>

using namespace std;

#define VKFFT_BACKEND 3
#define _CRT_SECURE_NO_WARNINGS 1

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#include <CL/cl.h>
#include "vkFFT.h"

// command line arguments
static int g_width = 128;
static int g_height = 128;
static int g_dump_result = false;

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    uint64_t device_id;
} VkGPU;


VkFFTResult init_device(VkGPU* vkGPU)
{
    cl_int res = CL_SUCCESS;
    cl_uint numPlatforms;

    res = clGetPlatformIDs(0, 0, &numPlatforms);
    if (res != CL_SUCCESS) 
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    if (!platforms) 
        return VKFFT_ERROR_MALLOC_FAILED;

    res = clGetPlatformIDs(numPlatforms, platforms, 0);
    if (res != CL_SUCCESS) 
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;

    uint64_t k = 0;
    for (uint64_t j = 0; j < numPlatforms; j++) {
        cl_uint numDevices;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
        cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        if (!deviceList) 
            return VKFFT_ERROR_MALLOC_FAILED;

        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
        if (res != CL_SUCCESS) 
            return VKFFT_ERROR_FAILED_TO_GET_DEVICE;

        for (uint64_t i = 0; i < numDevices; i++) {
            if (k == vkGPU->device_id) {
                vkGPU->platform = platforms[j];
                vkGPU->device = deviceList[i];
                vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
                if (res != CL_SUCCESS) 
                    return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;

                cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
                if (res != CL_SUCCESS) 
                    return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

                vkGPU->commandQueue = commandQueue;
                k++;
            } else {
                k++;
            }
        }
        free(deviceList);
    }
    free(platforms);

    return VKFFT_SUCCESS;
}

VkFFTResult fft_2d(VkGPU* vkGPU, int w, int h)
{
    cl_int res = CL_SUCCESS;
    //zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D
    configuration.size[0] = w;
    configuration.size[1] = h;
    configuration.numberBatches = 0;
    configuration.performR2C = 1; // perform R2C/C2R decomposition (0 - off, 1 - on)
    uint64_t num_items = configuration.size[0] * configuration.size[1];

    // out-of-place R2C FFT with custom strides
    uint64_t inputBufferSize = sizeof(float) * num_items;
    uint64_t outputBufferSize = sizeof(float) * 2 * num_items;
    uint64_t bufferSize = sizeof(float) * 2 * num_items;
    configuration.isInputFormatted = 1;

    configuration.inputBufferStride[0] = configuration.size[0];
    configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * configuration.size[1];
    //configuration.bufferStride[0] = (configuration.size[0] / 2) + 1;
    //configuration.bufferStride[1] = configuration.bufferStride[0] * configuration.size[1];

    vector<float> indata(num_items, 0);
    vector<float> outdata(2 * num_items, 0);
    for (size_t i = 0; i < num_items; i++) {
        indata[i] = i;
    }

    configuration.device = &vkGPU->device;
    configuration.platform = &vkGPU->platform;
    configuration.context = &vkGPU->context;

    // input buffer in device
    cl_mem clinbuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, inputBufferSize, 0, &res);
    if (res != CL_SUCCESS) 
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, clinbuffer, CL_TRUE, 0, inputBufferSize, indata.data(), 0, NULL, NULL);
    if (res != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_COPY;

    // computation buffer in device
    cl_mem clbuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
    if (res != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;

    // output buffer in device
    cl_mem cloutbuffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, outputBufferSize, 0, &res);
    if (res != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;

    configuration.inputBuffer = &clinbuffer;
    configuration.inputBufferSize = &inputBufferSize;
    configuration.buffer = &clbuffer;
    configuration.bufferSize = &bufferSize;
    //configuration.outputBuffer = &cloutbuffer;
    //configuration.outputBufferSize = &outputBufferSize;


    //configuration is initialized like in other examples
    //configuration.saveApplicationToString = 1;
    //configuration.loadApplicationFromString = 1; //choose one to save / load binary file

    if (configuration.loadApplicationFromString) {
        FILE* kernelCache;
        uint64_t str_len;

        kernelCache = fopen("VkFFT_binary", "r");
        fseek(kernelCache, 0, SEEK_END);
        str_len = ftell(kernelCache);
        fseek(kernelCache, 0, SEEK_SET);
        configuration.loadApplicationString = malloc(str_len);
        fread(configuration.loadApplicationString, str_len, 1, kernelCache);
        fclose(kernelCache);
    }


    VkFFTResult resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS && resFFT != VKFFT_ERROR_ENABLED_saveApplicationToString) {
        printf("ERROR: initializeVkFFT failed with resFFT = %d\n", resFFT);
        return resFFT;
    }

    if (configuration.saveApplicationToString) {
        FILE* kernelCache;
        kernelCache = fopen("VkFFT_binary", "w");
        fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
        fclose(kernelCache);
    }

    VkFFTLaunchParams launchParams = {};
    launchParams.inputBuffer = &clinbuffer;
    launchParams.buffer = &clbuffer;
    //launchParams.outputBuffer = &cloutbuffer;
    launchParams.commandQueue = &vkGPU->commandQueue;

    // FFT
    resFFT = VkFFTAppend(&app, -1, &launchParams);
    if (resFFT != VKFFT_SUCCESS) {
        printf("ERROR: FFT failed with resFFT = %d\n", resFFT);
        return resFFT;
    }
    clFinish(vkGPU->commandQueue);

    res = clEnqueueReadBuffer(vkGPU->commandQueue, clbuffer, CL_TRUE, 0, bufferSize, outdata.data(), 0, NULL, NULL);
    if (res != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_COPY;
    clFinish(vkGPU->commandQueue);
    if (g_dump_result) {
        ofstream outfile("result.txt");
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < 2*w; x++) {
                outfile << outdata[y*2*w + x] << ", ";
            }
            outfile << "\n";
        }
        //for (size_t i = 0; i < 2 * num_items; i++) {
        //    if (i % 16 == 0) printf("\n");
        //    printf("%f, ", outdata[i]);
        //}
        //printf("\n");
    }

#if 0
    // IFFT
    resFFT = VkFFTAppend(&app, 1, &launchParams);
    if (resFFT != VKFFT_SUCCESS) {
        printf("ERROR: IFFT failed with resFFT = %d\n", resFFT);
        return resFFT;
    }
    clFinish(vkGPU->commandQueue);

    res = clEnqueueReadBuffer(vkGPU->commandQueue, clbuffer, CL_TRUE, 0, bufferSize, outdata.data(), 0, NULL, NULL);
    if (res != CL_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_COPY;
    clFinish(vkGPU->commandQueue);

    for (size_t i = 0; i < 2 * num_items; i++) {
        if (i % 16 == 0) printf("\n");
        printf("%f, ", outdata[i]/256);
    }
    printf("\n");
#endif

    if (configuration.loadApplicationFromString)
        free(configuration.loadApplicationString);

    clReleaseMemObject(clbuffer);
    clReleaseMemObject(clinbuffer);
    clReleaseMemObject(cloutbuffer);
    deleteVkFFT(&app);

    return VKFFT_SUCCESS;
}

void parse_arg(int argc, char** argv)
{
    if (argc >= 3) {
        g_width = atoi(argv[1]);
        g_height = atoi(argv[2]);
    }

    for (size_t i = 1; i < argc; i++) {
        if (string(argv[i]) == "-d")
            g_dump_result = true;
    }
}

int main(int argc, char** argv)
{
    parse_arg(argc, argv);

    VkFFTResult resFFT = VKFFT_SUCCESS;
    VkGPU vkGPU = {};
    init_device(&vkGPU);

    resFFT = fft_2d(&vkGPU, g_width, g_height);
    printf("resFFT = % d\n", resFFT);

    return 0;
}