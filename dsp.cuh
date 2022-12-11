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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <memory>

const int num_streams = 4;
const int cal_mat_size = 16;
const int cal_mat_side = 4;

typedef thrust::complex<float> complex;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::device_vector<complex> gpuvec_c;
typedef thrust::host_vector<float> hostvec;
typedef thrust::host_vector<thrust::complex<float>> hostvec_c;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;
typedef thrust::device_vector<char> buffer_t;
typedef thrust::device_vector<char16_t> gpubuf;
typedef char *hostbuf;

// Returns the pointer from the thrust vector for using with Cuda Libraries
template <typename T>
inline T *get(thrust::device_vector<T> vec)
{
    return thrust::raw_pointer_cast(&vec[0]);
}

// The functor that applies to a trace to perform calibration
struct calibration_functor : public thrust::unary_function<complex, complex>
{
    const float qi, qq;
    const complex offsets;

    calibration_functor(float _qi, float _qq,
                        complex _offsets) : qi(_qi), qq(_qq),
                                            offsets(_offsets) {}

    __host__ __device__ complex operator()(const complex &x) const
    {
        complex x1 = x - offsets;
        return complex(x1.real(), qi * x1.real() + qq * x1.imag());
    }
};

class dsp
{
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_buf[num_streams];  // buffers for loading data
    gpubuf gpu_buf2[num_streams]; // buffers for loading data
    gpuvec_c data[num_streams];
    gpuvec_c data_calibrated[num_streams];
    gpuvec_c noise[num_streams];
    gpuvec_c noise_calibrated[num_streams];
    gpuvec_c data_complex[num_streams];

    gpuvec power[num_streams];   // arrays for storage of average power
    gpuvec_c field[num_streams]; // arrays for storage of average field
    gpuvec_c out[num_streams];

    int cnt = 0;

    /* Filtering window */
    gpuvec_c firwin;

    /* Subtraction trace */
    gpuvec_c subtraction_trace;

private:
    /* Useful variables */
    int trace_length; // for keeping the length of a trace
    int trace1_start, trace2_start, pitch;
    int batch_size;   // for keeping the number of segments in data array
    int total_length; // batch_size * trace_length
    int out_size;
    int semaphore = 0;             // for selecting the current stream
    complex scale = 500.f / 128.f; // for conversion into mV

    /* Streams' arrays */
    cudaStream_t streams[num_streams];
    thrust::cuda_cub::par_t::stream_attachment_type exec_policies[num_streams];
    NppStreamContext streamContexts[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];
    cublasHandle_t cublas_handles2[num_streams];

    /* Down-conversion variables */
    gpuvec_c downconversion_coeffs;

    /* Calibration variables */
    std::unique_ptr<calibration_functor> cal_fun;

    int batch_count;
    float alpha = 1.f;
    float beta = 0.f;

public:
    dsp(int len, int n, float part);

    ~dsp();

    /* The main function that receives data from the FIFO and
    runs the algorithm */
    void compute(const hostbuf buffer);

    /* Create a pinned memory buffer on the host for the FIFO to fill in */
    void createBuffer(size_t size)
    {
        this->handleError(cudaMallocHost((void **)&buffer, size));
    };

    /* Free up host buffer memory */
    void deleteBuffer() { this->handleError(cudaFreeHost(buffer)); };

    /* Get the pointer to the host buffer   */
    char *getBufferPointer() { return buffer; };

    /* Return a segment length */
    int getTraceLength() { return trace_length; };

    /* Return length of the whole batch */
    int getTotalLength() { return total_length; };

    /* Return size of the result storage */
    int getOutSize() { return out_size; };

    /* Return number of iterations completed */
    int getCounter() { return cnt; }

    /* Return the computed average field trace */
    void getCumulativeField(hostvec_c &result);

    /* Return the computed power trace */
    void getCumulativePower(hostvec &result);

    /* Return the computed correlation function */
    void getCorrelator(hostvec_c &result);

    /* Get the trace that will be subtracted from every measurement */
    void getSubtractionTrace(hostvec_c &trace) { trace = subtraction_trace; };

    /* Set the average field trace that will be subtracted
    from every measurement */
    void setSubtractionTrace(hostvec_c &trace) { subtraction_trace = trace; };

    /* Set the amplitude of the signal to scale it */
    void setAmplitude(int ampl) { scale = static_cast<float>(ampl) / 128.f; };

    /* Setup downconversion calibration parameters */
    void setDownConversionCalibrationParameters(float r, float phi,
                                                float offset_i, float offset_q);

    /* Set the IQ mixer intermeadiate frequency
    for digital down-conversion to dc */
    void setIntermediateFrequency(float frequency, int oversampling);

    /* Setup the filter parameters */
    void setFirwin(float cutoff_l, float cutoff_r, int oversampling = 1);

    /* Clear the subtraction trace */
    void resetSubtractionTrace()
    {
        thrust::fill(subtraction_trace.begin(), subtraction_trace.end(), 0);
    };

    /* Clear the result storage */
    void resetOutput()
    {
        for (int i = 0; i < num_streams; i++)
        {
            thrust::fill(out[i].begin(), out[i].end(), 0.f);
            thrust::fill(field[i].begin(), field[i].end(), 0.f);
            thrust::fill(power[i].begin(), power[i].end(), 0.f);
        }
    };

protected:
    void handleError(cudaError_t error);

    void switchStream()
    {
        semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0;
    };

    void loadDataToGPUwithPitchAndOffset(const hostbuf h_buf, gpubuf g_buf,
                                         size_t pitch, size_t offset,
                                         int stream_num);

    void convertDataToMilivolts(gpuvec_c data, gpubuf buf, int stream_num);

    void downconvert(gpuvec_c data, int stream_num);

    void applyDownConversionCalibration(gpuvec_c data, int stream_num);

    void addDataToOutput(gpuvec_c data, gpuvec_c output, int stream_num);

    void subtractDataFromOutput(gpuvec_c data, gpuvec_c output, int stream_num);

    void applyFilter(gpuvec_c data, const gpuvec_c window, int stream_num);

    void getMinMax(gpuvec data, int stream_num);

    void calculateField(int stream_num);

    void calculatePower(int stream_num);

    void calculateG1(int stream_num);
};

#endif // CPPMEASUREMENT_DSP_CUH