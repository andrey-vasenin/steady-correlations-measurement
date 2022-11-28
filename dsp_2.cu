//
// Created by andrei on 3/27/21.
//

#include "dsp.cuh"
#include <cstdio>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <npp.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <npps.h>
#include <complex>
#include <cublas_v2.h>
#include <cmath>

// DSP constructor
dsp::dsp(int len, int n)
{
    trace_length = len;
    batch_size = n;
    total_length = batch_size * trace_length;

    // allocating down-conversion calibration vectors
    this->handleError(cudaMalloc(&A_gpu, 2 * total_length * sizeof(float)));
    this->handleError(cudaMalloc(&C_gpu, total_length * sizeof(Npp32fc)));

    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        this->handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        auto nppStatus = nppGetStreamContext(&streamContexts[i]);
        if (nppStatus != NPP_SUCCESS)
            throw std::runtime_error("Couldn't get stream context");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_buf[i] = nppsMalloc_8s(2 * total_length);
        data[i] = nppsMalloc_32f(2 * total_length);
        out[i] = nppsMalloc_32f(2 * total_length);
        data_calibrated[i] = nppsMalloc_32f(2 * total_length);

        // Initialize the result array with zeros
        nppsSet_32f(0.f, out[i], 2 * total_length);

        // Initialize cuFFT plans
        auto cufft_err = cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        cufft_err = cufftSetStream(plans[i], streams[i]);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuFFT plan\n");

        // Initialize cuBLAS
        auto cublas_err = cublasCreate(&cublas_handles[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error initializing a cuBLAS handle\n");

        // Assign streams to cuBLAS handles
        cublas_err = cublasSetStream(cublas_handles[i], streams[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuBLAS handle\n");
    }
}

dsp::dsp(int len, int n, float part)
{
    trace_length = len * part;
    batch_size = n;
    total_length = batch_size * trace_length;

    trace1_start = len / 2 - trace_length - 1;
    trace2_start = len - trace_length - 1;
    pitch = len;

    // allocating down-conversion calibration vectors
    this->handleError(cudaMalloc(&A_gpu, 2 * total_length * sizeof(float)));
    this->handleError(cudaMalloc(&C_gpu, total_length * sizeof(Npp32fc)));

    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        this->handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        auto nppStatus = nppGetStreamContext(&streamContexts[i]);
        if (nppStatus != NPP_SUCCESS)
            throw std::runtime_error("Couldn't get stream context");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_buf[i] = nppsMalloc_8s(2 * total_length);
        gpu_buf2[i] = nppsMalloc_8s(2 * total_length);
        data[i] = nppsMalloc_32f(2 * total_length);
        noise[i] = nppsMalloc_32f(2 * total_length);
        data_calibrated[i] = nppsMalloc_32f(2 * total_length);
        noise_calibrated[i] = nppsMalloc_32f(2 * total_length);
        out[i] = nppsMalloc_32f(2 * total_length);

        // Initialize the result array with zeros
        nppsSet_32f(0.f, out[i], 2 * total_length);

        // Initialize cuFFT plans
        auto cufft_err = cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        cufft_err = cufftSetStream(plans[i], streams[i]);
        if (cufft_err != CUFFT_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuFFT plan\n");

        // Initialize cuBLAS
        auto cublas_err = cublasCreate(&cublas_handles[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error initializing a cuBLAS handle\n");

        // Assign streams to cuBLAS handles
        cublas_err = cublasSetStream(cublas_handles[i], streams[i]);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error assigning a stream to a cuBLAS handle\n");
    }
}

// DSP destructor
dsp::~dsp()
{
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        cublasDestroy(cublas_handles[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);

        // Destroy GPU streams
        this->handleError(cudaStreamDestroy(streams[i]));

        // Free GPU memory
        nppsFree(data[i]);
        nppsFree(data_complex[i]);
        nppsFree(data_calibrated[i]);
        nppsFree(gpu_buf[i]);
        nppsFree(out[i]);
        if (gpu_buf2[i] != nullptr)
            nppsFree(gpu_buf2[i]);
        if (noise[i] != nullptr)
            nppsFree(noise[i]);
        if (noise_calibrated[i] != nullptr)
            nppsFree(noise_calibrated[i]);
    }
    cudaFree(A_gpu);
    cudaFree(C_gpu);

    cudaDeviceReset();
}

int dsp::getTraceLength()
{
    return trace_length;
}

int dsp::getTotalLength()
{
    return total_length;
}

// Initializes matrices and arrays required for down-conversion calibration with given parameters
void dsp::setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q)
{
    // Filling A-matrix (4x4) in Fortran-style row order
    float a_ii = 1;
    float a_qi = std::tan(phi);
    float a_qq = 1 / (r * std::cos(phi));
    float A_mat[cal_mat_size];
    if (cal_mat_size == 4)
    {
        A_mat[0] = a_ii;
        A_mat[1] = a_qi;
        A_mat[2] = 0.f;
        A_mat[3] = a_qq;
    }
    else
    {
        for (int i = 0; i < cal_mat_size; i++)
            A_mat[i] = 0.f;
        A_mat[0] = a_ii;
        A_mat[1] = a_qi;
        A_mat[5] = a_qq;
        A_mat[10] = a_ii;
        A_mat[11] = a_qi;
        A_mat[15] = a_qq;
    }

    // Creating an array with repeated matrices
    std::vector<float> A_host(2 * total_length);
    for (int i = 0; i < 2 * total_length; i += cal_mat_size)
        for (int j = 0; j < cal_mat_size; j++)
            A_host[i + j] = A_mat[j];

    // Transferring it onto GPU memory
    this->handleError(cudaMemcpyAsync(A_gpu, A_host.data(), A_host.size() * sizeof(float),
        cudaMemcpyHostToDevice, streams[0]));

    // Estimating the number of matrix multiplications
    batch_count = 2 * total_length / cal_mat_size;

    // Filling the offsets array C_gpu
    Npp32fc init_val{ offset_i, offset_q };
    nppsSet_32fc(init_val, C_gpu, total_length);

    // cleaning
    cudaDeviceSynchronize();
}

// Error handler
void dsp::handleError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::string name = cudaGetErrorName(err);
        std::string text = cudaGetErrorString(err);
        throw std::runtime_error(name + ": " + text);
    }
}

void dsp::resetOutput()
{
    for (int i = 0; i < num_streams; i++)
        nppsSet_32f(0.f, out[i], 2 * total_length);
}

// Upload measured traces to GPU
void dsp::loadDataToGPU(const char* buffer)
{
    this->handleError(cudaMemcpyAsync(gpu_buf[semaphore], buffer, 2 * size_t(total_length) * sizeof(Npp8s),
        cudaMemcpyHostToDevice, streams[semaphore]));
}

void dsp::loadDataToGPUwithPitchAndOffset(const char* buffer, Npp8s* gpu_buf, size_t pitch, size_t offset)
{
    size_t width = 2 * size_t(trace_length) * sizeof(Npp8s);
    size_t src_pitch = 2 * pitch * sizeof(Npp8s);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    auto err = cudaMemcpy2DAsync(gpu_buf, dst_pitch,
        buffer + shift, src_pitch, width, batch_size,
        cudaMemcpyHostToDevice, streams[semaphore]);
    this->handleError(err);
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMilivolts(Npp32f* data, Npp8s* gpu_buf)
{
    // convert from int8 to float32
    auto status = nppsConvert_8s32f_Ctx(gpu_buf, data,
        2 * total_length, streamContexts[semaphore]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("error converting int8 to float32 #" + std::to_string(status));
    }
    // reinterpret interleaved channel samples as interleaved real and imaginary floats (basically do nothing)
    data_complex[semaphore] = reinterpret_cast<Npp32fc*>(data);
    // multiply by a constant in order to convert into mV
    status = nppsMulC_32fc_I_Ctx(scale, data_complex[semaphore], total_length, streamContexts[semaphore]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("error when scaling to mV #" + std::to_string(status));
    }
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(Npp32f* data, Npp32f* data_calibrated)
{
    using namespace std::string_literals;

    // Subtract offsets
    data_complex[semaphore] = reinterpret_cast<Npp32fc*>(data);
    auto npp_status = nppsSub_32fc_I_Ctx(C_gpu, data_complex[semaphore], total_length, streamContexts[semaphore]);
    if (npp_status != NPP_SUCCESS)
    {
        throw std::runtime_error("error when shifting offsets #" + std::to_string(npp_status));
    }

    // Apply rotation
    auto cublas_status = cublasGemmStridedBatchedEx(cublas_handles[semaphore],
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        cal_mat_side, cal_mat_side, cal_mat_side,
        &alpha,
        A_gpu, CUDA_R_32F, cal_mat_side, cal_mat_size,
        reinterpret_cast<float*>(data), CUDA_R_32F, cal_mat_side, cal_mat_size,
        &beta,
        reinterpret_cast<float*>(data_calibrated), CUDA_R_32F, cal_mat_side,
        cal_mat_size,
        batch_count,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        //                              CUBLAS_GEMM_ALGO13_TENSOR_OP);
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Error of batched matrix multiplication with code #"s
            + std::to_string(cublas_status));
    }
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(Npp32f* data)
{
    auto status = nppsAdd_32f_I_Ctx(data, out[semaphore],
        total_length, streamContexts[semaphore]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error adding new data to previous #" + std::to_string(status));
    }
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(Npp32f* data)
{
    auto status = nppsSub_32f_I_Ctx(data, out[semaphore],
        total_length, streamContexts[semaphore]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error adding new data to previous #" + std::to_string(status));
    }
}

// Computes field, i. e. <a(t)>
void dsp::computeField(const char* buffer)
{
    this->loadDataToGPU(buffer);
    this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
    this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
    this->addDataToOutput(data_calibrated[semaphore]);
    this->switchStream();
}

// Computes field, i. e. <a(t)>
void dsp::computeFieldAndNoise(const char* buffer, bool choose_noise)
{
    if (!choose_noise)
    {
        this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[semaphore], pitch, trace1_start);
        this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
        this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
        this->addDataToOutput(data_calibrated[semaphore]);
    }
    else
    {
        this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[semaphore], pitch, trace2_start);
        this->convertDataToMilivolts(noise[semaphore], gpu_buf2[semaphore]);
        this->applyDownConversionCalibration(noise[semaphore], noise_calibrated[semaphore]);
        this->addDataToOutput(noise_calibrated[semaphore]);
    }
    this->switchStream();
}

// Computes power P1, i. e. <a^dagger(t) a(t)>
void dsp::computePower(const char* buffer)
{
    this->loadDataToGPU(buffer);
    this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
    this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
    auto status = nppsSqr_32f_I_Ctx(data_calibrated[semaphore], 2 * total_length, streamContexts[semaphore]);
    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(status));
    }
    this->addDataToOutput(data_calibrated[semaphore]);
    this->switchStream();
}

// Computes power density, i. e. <a^dagger(omega) a(omega)>
void dsp::computePowerDensity(const char* buffer)
{
    this->loadDataToGPU(buffer);
    this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
    this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(data_calibrated[semaphore]);
    cufftExecC2C(plans[semaphore], cufft_data, cufft_data, CUFFT_FORWARD);
    auto npp_status = nppsSqr_32f_I_Ctx(data_calibrated[semaphore], 2 * total_length, streamContexts[semaphore]);
    if (npp_status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(npp_status));
    }
    this->addDataToOutput(data_calibrated[semaphore]);
    this->switchStream();
}

// Computes noise power density, i. e. <h^dagger(omega) h(omega)>
void dsp::computeNoisePowerDensity(const char* buffer)
{
    this->loadDataToGPU(buffer);
    this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
    this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(data_calibrated[semaphore]);
    cufftExecC2C(plans[semaphore], cufft_data, cufft_data, CUFFT_FORWARD);
    auto npp_status = nppsSqr_32f_I_Ctx(data_calibrated[semaphore], 2 * total_length, streamContexts[semaphore]);
    if (npp_status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(npp_status));
    }
    this->subtractDataFromOutput(data_calibrated[semaphore]);
    this->switchStream();
}

void dsp::computePowerDensityWithUltcalib(const char* buffer)
{
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[semaphore], pitch, trace1_start);
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[semaphore], pitch, trace2_start);
    this->convertDataToMilivolts(data[semaphore], gpu_buf[semaphore]);
    this->applyDownConversionCalibration(data[semaphore], data_calibrated[semaphore]);
    this->convertDataToMilivolts(noise[semaphore], gpu_buf2[semaphore]);
    this->applyDownConversionCalibration(noise[semaphore], noise_calibrated[semaphore]);
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(data_calibrated[semaphore]);
    cufftExecC2C(plans[semaphore], cufft_data, cufft_data, CUFFT_FORWARD);
    auto npp_status = nppsSqr_32f_I_Ctx(data_calibrated[semaphore], 2 * total_length, streamContexts[semaphore]);
    if (npp_status != NPP_SUCCESS) 
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(npp_status));
    }
    cufft_data = reinterpret_cast<cufftComplex*>(noise_calibrated[semaphore]);
    cufftExecC2C(plans[semaphore], cufft_data, cufft_data, CUFFT_FORWARD);
    npp_status = nppsSqr_32f_I_Ctx(noise_calibrated[semaphore], 2 * total_length, streamContexts[semaphore]);
    if (npp_status != NPP_SUCCESS)
    {
        throw std::runtime_error("Error squaring data #" + std::to_string(npp_status));
    }
    this->addDataToOutput(data_calibrated[semaphore]);
    this->subtractDataFromOutput(noise_calibrated[semaphore]);
    this->switchStream();
}

// Returns the average value
void dsp::getAverage(std::vector <std::complex<float>>& result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        nppsAdd_32f_I(out[i], out[0], total_length);
    this->handleError(cudaMemcpy(result.data(), out[0],
        2 * total_length * sizeof(Npp32f), cudaMemcpyDeviceToHost));
}