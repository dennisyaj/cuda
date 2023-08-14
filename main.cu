#include <iostream>
#include <cuda.h>

#define NUM_ELEMENTOS 1024
//-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

//--kernel de la suma
__global__
void addVector(float *d_A, float *d_B, float *d_C, int size) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index < size){
        d_C[index] = d_A[index] + d_B[index];}
}

int main() {
    float *h_A;
    float *h_B;
    float *h_C;

    float *d_A;
    float *d_B;
    float *d_C;

    int size_bytes = NUM_ELEMENTOS * sizeof(float);

    //-- solicitar memoria en el host
    h_A = (float *) malloc(size_bytes);
    h_B = (float *) malloc(size_bytes);
    h_C = (float *) malloc(size_bytes);


    //-- solicitar memoria en la gpu
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    //---inicializar lo vectores
    memset(h_C, 0, size_bytes);
    for (int i = 0; i < NUM_ELEMENTOS; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    //--copiar del host to device
    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice);

    //-- ejecutar el kernel
    addVector<<<4, 256>>>(d_A, d_B, d_C, NUM_ELEMENTOS);

    auto error = cudaGetLastError();
    if (error!= cudaSuccess){
        std::printf("ERROR: %s\n", cudaGetErrorString(error));
    }

    //--copiar del device to host
    cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost);

    //--imprimir le resultado
    std::printf("la suma es:\n ");
    for (int i = 0; i < NUM_ELEMENTOS; i++) {
        std::printf("%.0f, ", h_C[i]);
        if ((i + 1) % 50 == 0){
            std::printf("\n");}
    }
    //--librerar los recursos del host
    free(h_A);
    free(h_B);
    free(h_C);

    //--librerar los recursos del DEVICE
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;

}
