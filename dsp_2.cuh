//
// Created by andrei on 3/27/21.
//

#ifndef CPPMEASUREMENT_DSP_CUH
#define CPPMEASUREMENT_DSP_CUH

#include <nppdefs.h>
#include <vector>
#include <complex>
#include <cufft.h>
#include <cublas_v2.h>

const int num_streams = 4;
const int cal_mat_size = 16;
const int cal_mat_side = 4;

class dsp {
    /* Pointers to arrays with data */
    Npp8s* gpu_buf[num_streams];  // buffers for loading data
    Npp8s* gpu_buf2[num_streams];  // buffers for loading data
    Npp32f* data[num_streams];
    Npp32f* data_calibrated[num_streams];
    Npp32f* noise[num_streams];
    Npp32f* noise_calibrated[num_streams];
    Npp32fc* data_complex[num_streams];
    Npp32f* out[num_streams]; // arrays with results

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];

    /* Down-conversion calibration variables */
    float* A_gpu;
    Npp32fc* C_gpu;
    int batch_count;
    float alpha = 1.f;
    float beta = 0.f;

private:
    /* Useful variables */
    int trace_length; // for keeping the length of a trace
    int trace1_start, trace2_start, pitch;
    int batch_size; // for keeping the number of segments in data array
    int total_length; // batch_size * trace_length
    int semaphore = 0; // for selecting the current stream
    Npp32fc scale = Npp32fc{ 200.f / 128.f, 0.f }; // for conversion into mV

    /* Streams' arrays */
    cudaStream_t streams[num_streams];
    NppStreamContext streamContexts[num_streams];

public:
    dsp(int len, int n);

    dsp(int len, int n, float part);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    void resetOutput();

    void computeField(const char* buffer);

    void computeFieldAndNoise(const char* buffer, bool choose_noise);

    void computePower(const char* buffer);

    void computePowerDensity(const char* buffer);

    void computeNoisePowerDensity(const char* buffer);

    void computePowerDensityWithUltcalib(const char* buffer);

    void getAverage(std::vector <std::complex<float>>& result);

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

protected:
    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void loadDataToGPU(const char* buffer);

    void loadDataToGPUwithPitchAndOffset(const char* buffer, Npp8s* gpu_buf, size_t pitch, size_t offset);

    void convertDataToMilivolts(Npp32f* data, Npp8s* gpu_buf);

    void applyDownConversionCalibration(Npp32f* data, Npp32f* data_calibrated);

    void addDataToOutput(Npp32f* data);

    void subtractDataFromOutput(Npp32f* data);
};

#endif //CPPMEASUREMENT_DSP_CUH
