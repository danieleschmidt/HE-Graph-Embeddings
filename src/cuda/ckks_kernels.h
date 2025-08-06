#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Memory Management
cudaError_t cuda_allocate_memory(void** ptr, size_t size);
cudaError_t cuda_free_memory(void* ptr);
cudaError_t cuda_copy_to_device(void* dst, const void* src, size_t size);
cudaError_t cuda_copy_to_host(void* dst, const void* src, size_t size);

// CUDA Constants Initialization  
cudaError_t cuda_init_constants(const uint64_t* moduli, const uint64_t* roots, 
                               const uint64_t* inv_roots, const uint64_t* inv_n,
                               uint32_t num_moduli);

// NTT Operations
cudaError_t launch_ntt_forward(uint64_t* data, uint32_t n, uint32_t mod_idx,
                              cudaStream_t stream = 0);
cudaError_t launch_ntt_inverse(uint64_t* data, uint32_t n, uint32_t mod_idx,
                              cudaStream_t stream = 0);

// Polynomial Operations
cudaError_t launch_poly_mul_ntt(const uint64_t* a, const uint64_t* b, uint64_t* result,
                               uint32_t n, uint32_t mod_idx, cudaStream_t stream = 0);

// Homomorphic Encryption Operations
cudaError_t launch_he_add(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                         const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                         uint64_t* res_c0, uint64_t* res_c1,
                         uint32_t n, uint32_t mod_idx, cudaStream_t stream = 0);

cudaError_t launch_he_mul(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                         const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                         uint64_t* res_c0, uint64_t* res_c1, uint64_t* res_c2,
                         uint32_t n, uint32_t mod_idx, cudaStream_t stream = 0);

cudaError_t launch_rotate(const uint64_t* input, uint64_t* output,
                         uint32_t n, int32_t steps, uint32_t mod_idx,
                         cudaStream_t stream = 0);

cudaError_t launch_rescale(uint64_t* data_c0, uint64_t* data_c1, uint32_t n,
                          uint32_t old_level, uint32_t new_level, cudaStream_t stream = 0);

// Batch Operations
cudaError_t launch_batch_ntt_forward(uint64_t** data_batch, uint32_t batch_size,
                                    uint32_t n, uint32_t mod_idx, cudaStream_t stream = 0);

cudaError_t launch_batch_he_add(uint64_t** ct1_c0_batch, uint64_t** ct1_c1_batch,
                                uint64_t** ct2_c0_batch, uint64_t** ct2_c1_batch,
                                uint64_t** res_c0_batch, uint64_t** res_c1_batch,
                                uint32_t batch_size, uint32_t n, uint32_t mod_idx,
                                cudaStream_t stream = 0);

// Memory Utilities
cudaError_t launch_zero_memory(uint64_t* data, size_t size, cudaStream_t stream = 0);
cudaError_t launch_copy_memory(const uint64_t* src, uint64_t* dst, size_t size,
                              cudaStream_t stream = 0);

// Noise Estimation
cudaError_t launch_estimate_noise(const uint64_t* ciphertext, double* noise_estimate,
                                 uint32_t n, uint32_t level, double scale,
                                 cudaStream_t stream = 0);

// Stream Management
cudaError_t cuda_create_stream(cudaStream_t* stream);
cudaError_t cuda_destroy_stream(cudaStream_t stream);
cudaError_t cuda_stream_synchronize(cudaStream_t stream);

#ifdef __cplusplus
}
#endif