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

using namespace thrust::placeholders;

inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
    if (err != NPP_SUCCESS)
        throw std::runtime_error(msg);
}

// DSP constructor
dsp::dsp(int len, int n, float part) : trace_length{(int)std::round((float)len * part)}, // Length of a signal or noise trace
                                       batch_size{n},                                    // Number of segments in a buffer (same: number of traces in data)
                                       total_length{batch_size * trace_length},
                                       out_size{trace_length * trace_length},
                                       trace1_start{0},         // Start of the signal data
                                       trace2_start{len / 2},   // Start of the noise data
                                       pitch{len},              // Segment length in a buffer
                                       firwin(total_length, 1), // GPU memory for the filtering window
                                       subtraction_trace(total_length, 0),
                                       downconversion_coeffs(total_length, 1)
{
    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        check_npp_error(nppGetStreamContext(&streamContexts[i]), "Npp Error GetStreamContext");
        streamContexts[i].hStream = streams[i];
        exec_policies[i] = thrust::cuda::par.on(streams[i]);

        // Allocate arrays on GPU for every stream
        gpu_buf[i].resize(2 * total_length);
        gpu_buf2[i].resize(2 * total_length);
        data[i].resize(2 * total_length);
        noise[i].resize(2 * total_length);
        data_calibrated[i].resize(2 * total_length);
        noise_calibrated[i].resize(2 * total_length);
        power[i].resize(2 * total_length);
        field[i].resize(total_length);
        out[i].resize(out_size);

        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size),
                          "Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");

        // Initialize cuBLAS
        check_cublas_error(cublasCreate(&cublas_handles[i]),
                           "Error initializing a cuBLAS handle\n");
        check_cublas_error(cublasCreate(&cublas_handles2[i]),
                           "Error initializing a cuBLAS handle\n");

        // Assign streams to cuBLAS handles
        check_cublas_error(cublasSetStream(cublas_handles[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
        check_cublas_error(cublasSetStream(cublas_handles2[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
    }
    resetOutput();
    cal_fun = std::make_unique<calibration_functor>(0.f, 1.f, complex(0, 0));
}

// DSP destructor
dsp::~dsp()
{
    deleteBuffer();
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        cublasDestroy(cublas_handles[i]);
        cublasDestroy(cublas_handles2[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);

        // Destroy GPU streams
        handleError(cudaStreamDestroy(streams[i]));
    }
    cudaDeviceReset();
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
void dsp::setFirwin(float cutoff_l, float cutoff_r, int oversampling)
{
    hostvec_c hFirwin(total_length);
    float fs = 1250.f / (float)oversampling;
    int l_idx = (int)std::roundf((float)trace_length / fs * cutoff_l);
    int r_idx = (int)std::roundf((float)trace_length / fs * cutoff_r);
    ;
    for (int i = 0; i < total_length; i++)
    {
        int j = i % trace_length;
        hFirwin[i] = ((j < l_idx) || (j > r_idx)) ? 0.f : 1.0f;
    }
    firwin = hFirwin;
}

// Initializes matrices and arrays required for down-conversion calibration with given parameters
void dsp::setDownConversionCalibrationParameters(float r, float phi,
                                                 float offset_i, float offset_q)
{
    float a_qi = std::tan(phi);
    float a_qq = 1 / (r * std::cos(phi));
    cal_fun = std::make_unique<calibration_functor>(a_qi, a_qq,
                                                    complex(offset_i, offset_q));
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(gpuvec_c data, int stream_num)
{
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(), data.begin(), *cal_fun);
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

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    using namespace std::complex_literals;
    hostvec_c dcov_host(total_length);

    const float pi = std::acos(-1);

    float ovs = static_cast<float>(oversampling);
    float t = 0;

    for (int j = 0; j < batch_size; j++)
    {
        for (int k = 0; k < trace_length; k++)
        {
            t = 0.8 * k * ovs;
            dcov_host[j * trace_length + k] = std::exp(-2if * pi * frequency * t);
        }
    }
    downconversion_coeffs = dcov_host;
}

void dsp::downconvert(gpuvec_c data, int stream_num)
{
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(), downconversion_coeffs.begin(), data.begin(), _1 * _2);
}

void dsp::compute(const hostbuf buffer)
{
    const int stream_num = semaphore;
    this->switchStream();
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf[stream_num], pitch, trace1_start, stream_num);
    this->loadDataToGPUwithPitchAndOffset(buffer, gpu_buf2[stream_num], pitch, trace2_start, stream_num);
    this->convertDataToMilivolts(data[stream_num], gpu_buf[stream_num], stream_num);
    this->convertDataToMilivolts(noise[stream_num], gpu_buf2[stream_num], stream_num);
    this->applyDownConversionCalibration(data[stream_num], stream_num);
    this->applyDownConversionCalibration(noise[stream_num], stream_num);
    this->applyFilter(data[stream_num], firwin, stream_num);
    this->applyFilter(noise[stream_num], firwin, stream_num);
    this->downconvert(data[stream_num], stream_num);
    this->downconvert(noise[stream_num], stream_num);
    this->subtractDataFromOutput(subtraction_trace, data[stream_num], stream_num);

    this->calculateField(stream_num);
    this->calculateG1(stream_num);
    this->calculatePower(stream_num);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::loadDataToGPUwithPitchAndOffset(const hostbuf h_buf, gpubuf g_buf, size_t pitch, size_t offset, int stream_num)
{
    size_t width = 2 * size_t(trace_length) * sizeof(Npp8s);
    size_t src_pitch = 2 * pitch * sizeof(Npp8s);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    auto err = cudaMemcpy2DAsync(get(g_buf), dst_pitch,
                                 h_buf + shift, src_pitch, width, batch_size,
                                 cudaMemcpyHostToDevice, streams[stream_num]);
    this->handleError(err);
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMilivolts(gpuvec_c data, gpubuf gpu_buf, int stream_num)
{
    // convert from int8 to float32
    auto err = nppsConvert_8s32f_Ctx(reinterpret_cast<Npp8s *>(get(gpu_buf)),
                                     reinterpret_cast<Npp32f *>(get(data)),
                                     gpu_buf.size(),
                                     streamContexts[stream_num]);
    check_npp_error(err, "error converting int8 to float32 " + std::to_string(err));
    // multiply by a constant in order to convert into mV
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(),
                      data.begin(), _1 * scale);
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec_c data, const gpuvec_c window, int stream_num)
{
    // Step 1. Take FFT of each segment
    cufftComplex *cufft_data = reinterpret_cast<cufftComplex *>(get(data));
    auto cufft_status = cufftExecC2C(plans[stream_num], cufft_data,
                                     cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufft_status, "Error taking fourier " + std::to_string(cufft_status));

    // Step 2. Multiply each segment by a window
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(),
                      firwin.begin(), data.begin(), _1 * _2);

    // Step 3. Take inverse FFT of each segment
    cufft_status = cufftExecC2C(plans[stream_num], cufft_data,
                                cufft_data, CUFFT_INVERSE);
    check_cufft_error(cufft_status, "Error taking fourier " + std::to_string(cufft_status));

    // Step 4. Normalize the FFT for the output to equal the input
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(),
                      data.begin(), _1 / static_cast<float>(trace_length));
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(gpuvec_c data, gpuvec_c output, int stream_num)
{
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(),
                      output.begin(), output.begin(), _1 + _2);
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(gpuvec_c data, gpuvec_c output, int stream_num)
{
    thrust::transform(exec_policies[stream_num], data.begin(), data.end(),
                      output.begin(), output.begin(), _2 - _1);
}

// Calculates the field from the data in the GPU memory
void dsp::calculateField(int stream_num)
{
    // Add signal field to the cumulative field
    this->addDataToOutput(data[stream_num], field[stream_num], stream_num);
    // Subtract noise field from the cumulative field
    this->subtractDataFromOutput(noise[stream_num], field[stream_num],
                                 stream_num);
}

void dsp::calculateG1(int stream_num)
{
    using namespace std::string_literals;

    const float alpha_data = 1;   // this alpha multiplies the result to be added to the output
    const float alpha_noise = -1; // this alpha multiplies the result to be added to the output
    const float beta = 1;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCherk(cublas_handles2[stream_num],
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                     &alpha_data, reinterpret_cast<cuComplex *>(get(data[stream_num])), trace_length,
                                     &beta, reinterpret_cast<cuComplex *>(get(out[stream_num])), trace_length);
    // Check for errors
    check_cublas_error(cublas_status, "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
    // Compute correlation for the noise and subtract it from the output
    cublas_status = cublasCherk(cublas_handles2[stream_num],
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                &alpha_noise, reinterpret_cast<cuComplex *>(get(noise_calibrated[stream_num])), trace_length,
                                &beta, reinterpret_cast<cuComplex *>(get(out[stream_num])), trace_length);
    // Check for errors
    check_cublas_error(cublas_status, "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
}

// Returns the average value
void dsp::getCorrelator(hostvec_c &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(out[i].begin(), out[i].end(), out[0].begin(),
                          out[0].begin(), _1 + _2);
    result = out[0];
}

// Returns the cumulative power
void dsp::getCumulativePower(hostvec &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(power[i].begin(), power[i].end(), power[0].begin(),
                          power[0].begin(), _1 + _2);
    result = power[0];
}

// Returns the cumulative field
void dsp::getCumulativeField(hostvec_c &result)
{
    this->handleError(cudaDeviceSynchronize());
    for (int i = 1; i < num_streams; i++)
        thrust::transform(field[i].begin(), field[i].end(), field[0].begin(),
                          field[0].begin(), _1 + _2);
    result = field[0];
}
