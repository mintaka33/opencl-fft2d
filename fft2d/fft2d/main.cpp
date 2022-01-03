#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <inttypes.h>

using namespace std;

#define VKFFT_BACKEND 3
#define _CRT_SECURE_NO_WARNINGS 1

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#include <CL/cl.h>
#include "vkFFT.h"

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

VkFFTResult fft_2d(VkGPU* vkGPU)
{
    cl_int res = CL_SUCCESS;
    //zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D
    configuration.size[0] = 16;
    configuration.size[1] = 16;
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
    for (size_t i = 0; i < num_items; i++) {
        indata[i] = i % 256;
    }
    for (size_t i = 0; i < num_items; i++) {
        if (i % 16 == 0) printf("\n");
        printf("%f, ", indata[i]);
    }
    printf("\n");
    vector<float> outdata(2 * num_items, 0);

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
    VkFFTResult resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) {
        printf("ERROR: initializeVkFFT failed with resFFT = %d\n", resFFT);
        return resFFT;
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

    for (size_t i = 0; i < 2*num_items; i++) {
        if (i % 16 == 0) printf("\n");
        printf("%f, ", outdata[i]);
    }
    printf("\n");

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

    clReleaseMemObject(clbuffer);
    clReleaseMemObject(clinbuffer);
    clReleaseMemObject(cloutbuffer);
    deleteVkFFT(&app);

    return VKFFT_SUCCESS;
}

int main()
{
    VkFFTResult resFFT = VKFFT_SUCCESS;

    VkGPU vkGPU = {};
    init_device(&vkGPU);

    resFFT = fft_2d(&vkGPU);
    printf("resFFT = % d\n", resFFT);

    return 0;
}