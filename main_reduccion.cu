#include <iostream>
#include <cuda.h>
#include <vector>

#define NUM_ELEMENTOS 1024

using namespace std;

__global__
void gpu_reduccion(float *d_input, float *d_output) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (threadIdx.x % (s << 1 == 0)) {
            d_input[index] = d_input[index] + d_input[index + s];
        }
        //sincronizar
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_output[blockIdx.x] = d_input[index];

}

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void cuda_info() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int cores = getSPcores(prop);

        std::cout << "Device Number: " << i << std::endl;
        std::cout << " Device Name: " << prop.name << std::endl;
        std::cout << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << " Total Global Memory (Gbytes): " << (prop.totalGlobalMem/(1024*1024*1024)) << std::endl;
        std::cout << " Number of multiprocessors: " << prop.multiProcessorCount <<std::endl;
        std::cout << " Number of cores: " << cores <<std::endl;

        std::cout << " Max GridX: " << prop.maxGridSize[0] << std::endl;
        std::cout << " Max GridY: " << prop.maxGridSize[1] << std::endl;
        std::cout << " Max GridZ: " << prop.maxGridSize[2] << std::endl;
        std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        //std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;

        std::cout << " Memory Clock Rate (kHz): " << prop.memoryClockRate << std::endl;
        std::cout << " Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << " L2 Cache Size (bytes): " << prop.l2CacheSize << std::endl;
    }
}
int main() {

    cuda_info();

    int threads_per_block = 1024;
    int blocks_per_grid = NUM_ELEMENTOS / threads_per_block;
    vector<float> h_input(NUM_ELEMENTOS);
    vector<float> h_output(blocks_per_grid);

    float *d_input;
    float *d_output;
    float *d_suma;

    int size_bytes = NUM_ELEMENTOS * sizeof(float);

    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, blocks_per_grid * sizeof(float));
    cudaMalloc(&d_suma, 1 * sizeof(float));

    for (int i = 0; i < NUM_ELEMENTOS; i++) {
        h_input[i] = i + 1;
    }

    cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice);
    //lanzar el kernel
    gpu_reduccion<<<blocks_per_grid, threads_per_block>>>(d_input, d_output);
    gpu_reduccion<<<1, blocks_per_grid>>>(d_output, d_suma);
    float suma = 0;
    cudaMemcpy(&suma, d_suma, 1 * sizeof(float), cudaMemcpyDeviceToHost);

//    cudaMemcpy(h_output.data(), d_output, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
//
//    float suma = 0;
//    for (int i = 0; i < blocks_per_grid; i++) {
//        suma = suma + h_output[i];
//    }
    std::printf("suma %.0f\n", suma);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_suma);
}