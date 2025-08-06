#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            return error; \
        } \
    } while(0)

#define MAX_MODULI 32
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

namespace hegraph {
namespace cuda {

// Constants for NTT - Using 61-bit safe primes for CKKS
__constant__ uint64_t d_modulus[MAX_MODULI];     // Prime moduli
__constant__ uint64_t d_roots[MAX_MODULI];       // Primitive roots
__constant__ uint64_t d_inv_roots[MAX_MODULI];   // Inverse roots
__constant__ uint64_t d_inv_n[MAX_MODULI];       // Precomputed n^(-1) mod p

// Modular arithmetic functions
__device__ inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    // Fast modular multiplication using __umul64hi for 64-bit multiplication
    uint64_t hi, lo;
    lo = a * b;
    hi = __umul64hi(a, b);
    
    // Simple reduction - can be optimized with Barrett reduction
    if (hi == 0) {
        return lo >= mod ? lo % mod : lo;
    }
    
    // For high precision, use double precision division
    double quotient = (static_cast<double>(hi) * 4294967296.0 * 4294967296.0 + 
                      static_cast<double>(lo)) / static_cast<double>(mod);
    uint64_t q = static_cast<uint64_t>(quotient);
    
    uint64_t remainder = lo - q * mod;
    if (hi > 0 || remainder >= mod) {
        remainder = (remainder >= mod) ? remainder - mod : remainder;
    }
    
    return remainder;
}

__device__ inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t sum = a + b;
    return (sum >= mod) ? sum - mod : sum;
}

__device__ inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    return (a >= b) ? a - b : a + mod - b;
}

__device__ inline uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod);
        }
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// NTT butterfly operation
__device__ void butterfly(uint64_t& a, uint64_t& b, uint64_t w, uint64_t mod) {
    uint64_t t = mod_mul(b, w, mod);
    b = mod_sub(a, t, mod);
    a = mod_add(a, t, mod);
}

// Forward NTT kernel - Decimation in Time
__global__ void ntt_forward_kernel(uint64_t* data, uint32_t n, uint32_t log_n, 
                                  uint32_t mod_idx) {
    uint32_t tid = threadIdx.x;
    uint32_t global_idx = blockIdx.x * blockDim.x + tid;
    
    if (global_idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    uint64_t root = d_roots[mod_idx];
    
    // Bit-reverse permutation
    uint32_t reversed_idx = 0;
    for (uint32_t i = 0; i < log_n; ++i) {
        reversed_idx = (reversed_idx << 1) | ((global_idx >> i) & 1);
    }
    
    // Load data with bit-reversed indexing
    uint64_t temp = data[reversed_idx];
    __syncthreads();
    
    // Store back with correct indexing for NTT
    data[global_idx] = temp;
    __syncthreads();
    
    // Cooley-Tukey NTT stages
    for (uint32_t stage = 1; stage <= log_n; ++stage) {
        uint32_t m = 1 << stage;          // 2^stage
        uint32_t half_m = m >> 1;         // 2^(stage-1)
        
        // Calculate the principal root of unity for this stage
        uint64_t stage_root = mod_pow(root, n / m, mod);
        
        // Process butterflies for this thread
        uint32_t k = global_idx;
        uint32_t group = k / half_m;
        uint32_t pos_in_group = k % half_m;
        uint32_t partner_idx = group * m + pos_in_group + half_m;
        
        if (partner_idx < n) {
            uint64_t w = mod_pow(stage_root, pos_in_group, mod);
            
            uint64_t u = data[k];
            uint64_t v = mod_mul(data[partner_idx], w, mod);
            
            data[k] = mod_add(u, v, mod);
            data[partner_idx] = mod_sub(u, v, mod);
        }
        
        __syncthreads();
    }
}

// Inverse NTT kernel - Decimation in Frequency
__global__ void ntt_inverse_kernel(uint64_t* data, uint32_t n, uint32_t log_n,
                                  uint32_t mod_idx) {
    uint32_t tid = threadIdx.x;
    uint32_t global_idx = blockIdx.x * blockDim.x + tid;
    
    if (global_idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    uint64_t inv_root = d_inv_roots[mod_idx];
    
    // Gentleman-Sande INTT stages (reverse order)
    for (uint32_t stage = log_n; stage >= 1; --stage) {
        uint32_t m = 1 << stage;          // 2^stage
        uint32_t half_m = m >> 1;         // 2^(stage-1)
        
        // Calculate the principal inverse root of unity for this stage
        uint64_t stage_inv_root = mod_pow(inv_root, n / m, mod);
        
        // Process butterflies for this thread
        uint32_t k = global_idx;
        uint32_t group = k / half_m;
        uint32_t pos_in_group = k % half_m;
        uint32_t partner_idx = group * m + pos_in_group + half_m;
        
        if (partner_idx < n) {
            uint64_t u = data[k];
            uint64_t v = data[partner_idx];
            
            data[k] = mod_add(u, v, mod);
            
            uint64_t diff = mod_sub(u, v, mod);
            uint64_t w = mod_pow(stage_inv_root, pos_in_group, mod);
            data[partner_idx] = mod_mul(diff, w, mod);
        }
        
        __syncthreads();
    }
    
    // Scale by n^(-1)
    uint64_t n_inv = d_inv_n[mod_idx];
    data[global_idx] = mod_mul(data[global_idx], n_inv, mod);
    
    __syncthreads();
    
    // Bit-reverse permutation for final output
    uint32_t reversed_idx = 0;
    for (uint32_t i = 0; i < log_n; ++i) {
        reversed_idx = (reversed_idx << 1) | ((global_idx >> i) & 1);
    }
    
    // Store with bit-reversed indexing
    uint64_t temp = data[global_idx];
    __syncthreads();
    
    data[reversed_idx] = temp;
}

// Polynomial multiplication in NTT domain
__global__ void poly_mul_ntt_kernel(const uint64_t* a, const uint64_t* b,
                                   uint64_t* result, uint32_t n, uint32_t mod_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    result[idx] = mod_mul(a[idx], b[idx], mod);
}

// Homomorphic addition kernel
__global__ void he_add_kernel(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                             const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                             uint64_t* res_c0, uint64_t* res_c1,
                             uint32_t n, uint32_t mod_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    res_c0[idx] = mod_add(ct1_c0[idx], ct2_c0[idx], mod);
    res_c1[idx] = mod_add(ct1_c1[idx], ct2_c1[idx], mod);
}

// Homomorphic multiplication kernel (tensor product)
__global__ void he_mul_kernel(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                             const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                             uint64_t* res_c0, uint64_t* res_c1, uint64_t* res_c2,
                             uint32_t n, uint32_t mod_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    
    // c0 = a0 * b0
    res_c0[idx] = mod_mul(ct1_c0[idx], ct2_c0[idx], mod);
    
    // c1 = a0 * b1 + a1 * b0
    uint64_t t1 = mod_mul(ct1_c0[idx], ct2_c1[idx], mod);
    uint64_t t2 = mod_mul(ct1_c1[idx], ct2_c0[idx], mod);
    res_c1[idx] = mod_add(t1, t2, mod);
    
    // c2 = a1 * b1
    res_c2[idx] = mod_mul(ct1_c1[idx], ct2_c1[idx], mod);
}

// Rescaling kernel with proper RNS base conversion
__global__ void rescale_kernel(uint64_t* data_c0, uint64_t* data_c1, uint32_t n, 
                              uint32_t old_level, uint32_t new_level) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // For CKKS rescaling, we need to perform RNS base conversion
    // This is a simplified version - proper implementation requires
    // fast base conversion using Montgomery representation
    
    uint64_t last_mod = d_modulus[old_level];
    uint64_t last_mod_inv = mod_pow(last_mod, d_modulus[new_level] - 2, d_modulus[new_level]);
    
    // Convert from RNS representation with old_level+1 moduli to new_level+1 moduli
    // Simplified: just take modulo of the new level modulus
    if (new_level < old_level) {
        data_c0[idx] = data_c0[idx] % d_modulus[new_level];
        data_c1[idx] = data_c1[idx] % d_modulus[new_level];
    }
}

// Rotation kernel using Galois automorphism
__global__ void rotate_kernel(const uint64_t* input, uint64_t* output,
                             uint32_t n, int32_t steps, uint32_t mod_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Apply cyclic rotation
    uint32_t n_half = n >> 1;
    uint32_t rot_steps = ((steps % n_half) + n_half) % n_half;
    uint32_t new_idx = (idx + rot_steps) % n;
    
    output[new_idx] = input[idx];
}

// Bootstrap kernel (simplified placeholder)
__global__ void bootstrap_kernel(uint64_t* ciphertext, uint32_t n,
                                uint32_t mod_idx, double scale) {
    // Bootstrapping is extremely complex and would require
    // homomorphic evaluation of modular reduction and rounding
    // This is a placeholder for the actual implementation
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Placeholder: just refresh the modulus chain
    uint64_t mod = d_modulus[mod_idx];
    ciphertext[idx] = ciphertext[idx] % mod;
}

// Graph-specific kernels

// Encrypted GraphSAGE aggregation kernel
__global__ void he_graphsage_aggregate_kernel(
    const uint64_t* node_features_c0,
    const uint64_t* node_features_c1,
    const int32_t* edge_src,
    const int32_t* edge_dst,
    uint64_t* aggregated_c0,
    uint64_t* aggregated_c1,
    uint32_t num_edges,
    uint32_t feature_dim,
    uint32_t poly_degree,
    uint32_t mod_idx
) {
    uint32_t edge_id = blockIdx.x;
    uint32_t feat_id = threadIdx.x;
    
    if (edge_id >= num_edges || feat_id >= feature_dim) return;
    
    int32_t src = edge_src[edge_id];
    int32_t dst = edge_dst[edge_id];
    
    uint64_t mod = d_modulus[mod_idx];
    
    // Calculate indices
    uint32_t src_idx = src * feature_dim * poly_degree + feat_id * poly_degree;
    uint32_t dst_idx = dst * feature_dim * poly_degree + feat_id * poly_degree;
    
    // Aggregate features (simplified - actual implementation needs atomic operations)
    for (uint32_t i = 0; i < poly_degree; ++i) {
        atomicAdd(&aggregated_c0[dst_idx + i], node_features_c0[src_idx + i]);
        atomicAdd(&aggregated_c1[dst_idx + i], node_features_c1[src_idx + i]);
    }
}

// Encrypted attention mechanism kernel
__global__ void he_attention_kernel(
    const uint64_t* query_c0,
    const uint64_t* query_c1,
    const uint64_t* key_c0,
    const uint64_t* key_c1,
    const uint64_t* value_c0,
    const uint64_t* value_c1,
    uint64_t* output_c0,
    uint64_t* output_c1,
    uint32_t seq_len,
    uint32_t d_model,
    uint32_t poly_degree,
    uint32_t mod_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= seq_len * d_model) return;
    
    uint32_t pos = tid / d_model;
    uint32_t dim = tid % d_model;
    
    uint64_t mod = d_modulus[mod_idx];
    
    // Compute attention scores (simplified)
    // In practice, this requires polynomial approximation of softmax
    uint64_t score = 0;
    for (uint32_t i = 0; i < seq_len; ++i) {
        // Q * K^T computation
        uint32_t q_idx = pos * d_model * poly_degree + dim * poly_degree;
        uint32_t k_idx = i * d_model * poly_degree + dim * poly_degree;
        
        // Simplified dot product
        for (uint32_t j = 0; j < poly_degree; ++j) {
            score = mod_add(score, 
                          mod_mul(query_c0[q_idx + j], key_c0[k_idx + j], mod),
                          mod);
        }
    }
    
    // Apply attention to values (simplified)
    uint32_t out_idx = pos * d_model * poly_degree + dim * poly_degree;
    for (uint32_t j = 0; j < poly_degree; ++j) {
        output_c0[out_idx + j] = mod_mul(score, value_c0[out_idx + j], mod);
        output_c1[out_idx + j] = mod_mul(score, value_c1[out_idx + j], mod);
    }
}

// Batch processing kernels

// Batch NTT forward kernel
__global__ void batch_ntt_forward_kernel(uint64_t** data_batch, uint32_t batch_size,
                                        uint32_t n, uint32_t log_n, uint32_t mod_idx) {
    uint32_t batch_idx = blockIdx.y;
    uint32_t element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || element_idx >= n) return;
    
    uint64_t* data = data_batch[batch_idx];
    uint64_t mod = d_modulus[mod_idx];
    uint64_t root = d_roots[mod_idx];
    
    // Bit-reverse permutation
    uint32_t reversed_idx = 0;
    for (uint32_t i = 0; i < log_n; ++i) {
        reversed_idx = (reversed_idx << 1) | ((element_idx >> i) & 1);
    }
    
    uint64_t temp = data[reversed_idx];
    __syncthreads();
    data[element_idx] = temp;
    __syncthreads();
    
    // NTT computation
    for (uint32_t stage = 1; stage <= log_n; ++stage) {
        uint32_t m = 1 << stage;
        uint32_t half_m = m >> 1;
        uint64_t stage_root = mod_pow(root, n / m, mod);
        
        uint32_t k = element_idx;
        uint32_t group = k / half_m;
        uint32_t pos_in_group = k % half_m;
        uint32_t partner_idx = group * m + pos_in_group + half_m;
        
        if (partner_idx < n) {
            uint64_t w = mod_pow(stage_root, pos_in_group, mod);
            uint64_t u = data[k];
            uint64_t v = mod_mul(data[partner_idx], w, mod);
            
            data[k] = mod_add(u, v, mod);
            data[partner_idx] = mod_sub(u, v, mod);
        }
        __syncthreads();
    }
}

// Batch homomorphic operations
__global__ void batch_he_add_kernel(uint64_t** ct1_c0_batch, uint64_t** ct1_c1_batch,
                                   uint64_t** ct2_c0_batch, uint64_t** ct2_c1_batch,
                                   uint64_t** res_c0_batch, uint64_t** res_c1_batch,
                                   uint32_t batch_size, uint32_t n, uint32_t mod_idx) {
    uint32_t batch_idx = blockIdx.y;
    uint32_t element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || element_idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    
    res_c0_batch[batch_idx][element_idx] = mod_add(
        ct1_c0_batch[batch_idx][element_idx],
        ct2_c0_batch[batch_idx][element_idx], mod);
    
    res_c1_batch[batch_idx][element_idx] = mod_add(
        ct1_c1_batch[batch_idx][element_idx],
        ct2_c1_batch[batch_idx][element_idx], mod);
}

// Memory management kernels

// Zero memory kernel
__global__ void zero_memory_kernel(uint64_t* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0;
    }
}

// Copy memory kernel
__global__ void copy_memory_kernel(const uint64_t* src, uint64_t* dst, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Memory pool allocation kernel
__global__ void init_memory_pool(uint64_t* pool, size_t size, uint64_t init_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        pool[idx] = init_val;
    }
}

// Noise estimation kernel
__global__ void estimate_noise_kernel(const uint64_t* ciphertext,
                                     double* noise_estimate,
                                     uint32_t n, uint32_t level,
                                     double scale) {
    extern __shared__ double shared_noise[];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    shared_noise[tid] = 0.0;
    
    if (idx < n) {
        // Simplified noise estimation
        uint64_t mod = d_modulus[level];
        double normalized = static_cast<double>(ciphertext[idx]) / mod;
        shared_noise[tid] = normalized * normalized;
    }
    __syncthreads();
    
    // Reduction to compute variance
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_noise[tid] += shared_noise[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(noise_estimate, shared_noise[0] / n);
    }
}

// Host wrapper functions

extern "C" {

// Initialize CUDA constants
cudaError_t cuda_init_constants(const uint64_t* moduli, const uint64_t* roots, 
                               const uint64_t* inv_roots, const uint64_t* inv_n,
                               uint32_t num_moduli) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_modulus, moduli, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_roots, roots, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_inv_roots, inv_roots, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_inv_n, inv_n, num_moduli * sizeof(uint64_t)));
    return cudaSuccess;
}

// Memory management
cudaError_t cuda_allocate_memory(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t cuda_free_memory(void* ptr) {
    return cudaFree(ptr);
}

cudaError_t cuda_copy_to_device(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

cudaError_t cuda_copy_to_host(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// NTT operations
cudaError_t launch_ntt_forward(uint64_t* data, uint32_t n, uint32_t mod_idx,
                              cudaStream_t stream) {
    if (n == 0 || (n & (n - 1)) != 0) {
        return cudaErrorInvalidValue;  // n must be power of 2
    }
    
    uint32_t log_n = __builtin_ctz(n);
    uint32_t threads = min(n, static_cast<uint32_t>(MAX_THREADS_PER_BLOCK));
    uint32_t blocks = (n + threads - 1) / threads;
    
    ntt_forward_kernel<<<blocks, threads, 0, stream>>>(
        data, n, log_n, mod_idx
    );
    
    return cudaGetLastError();
}

cudaError_t launch_ntt_inverse(uint64_t* data, uint32_t n, uint32_t mod_idx,
                              cudaStream_t stream) {
    if (n == 0 || (n & (n - 1)) != 0) {
        return cudaErrorInvalidValue;  // n must be power of 2
    }
    
    uint32_t log_n = __builtin_ctz(n);
    uint32_t threads = min(n, static_cast<uint32_t>(MAX_THREADS_PER_BLOCK));
    uint32_t blocks = (n + threads - 1) / threads;
    
    ntt_inverse_kernel<<<blocks, threads, 0, stream>>>(
        data, n, log_n, mod_idx
    );
    
    return cudaGetLastError();
}

// Polynomial operations
cudaError_t launch_poly_mul_ntt(const uint64_t* a, const uint64_t* b, uint64_t* result,
                               uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    poly_mul_ntt_kernel<<<blocks, threads, 0, stream>>>(
        a, b, result, n, mod_idx
    );
    
    return cudaGetLastError();
}

// Homomorphic operations
cudaError_t launch_he_add(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                         const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                         uint64_t* res_c0, uint64_t* res_c1,
                         uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    he_add_kernel<<<blocks, threads, 0, stream>>>(
        ct1_c0, ct1_c1, ct2_c0, ct2_c1, res_c0, res_c1, n, mod_idx
    );
    
    return cudaGetLastError();
}

cudaError_t launch_he_mul(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                         const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                         uint64_t* res_c0, uint64_t* res_c1, uint64_t* res_c2,
                         uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    he_mul_kernel<<<blocks, threads, 0, stream>>>(
        ct1_c0, ct1_c1, ct2_c0, ct2_c1, res_c0, res_c1, res_c2, n, mod_idx
    );
    
    return cudaGetLastError();
}

cudaError_t launch_rotate(const uint64_t* input, uint64_t* output,
                         uint32_t n, int32_t steps, uint32_t mod_idx,
                         cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    rotate_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n, steps, mod_idx
    );
    
    return cudaGetLastError();
}

cudaError_t launch_rescale(uint64_t* data_c0, uint64_t* data_c1, uint32_t n,
                          uint32_t old_level, uint32_t new_level, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    rescale_kernel<<<blocks, threads, 0, stream>>>(
        data_c0, data_c1, n, old_level, new_level
    );
    
    return cudaGetLastError();
}

// Batch operations
cudaError_t launch_batch_ntt_forward(uint64_t** data_batch, uint32_t batch_size,
                                    uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    if (n == 0 || (n & (n - 1)) != 0) {
        return cudaErrorInvalidValue;
    }
    
    uint32_t log_n = __builtin_ctz(n);
    uint32_t threads = min(n, static_cast<uint32_t>(MAX_THREADS_PER_BLOCK));
    
    dim3 blocks((n + threads - 1) / threads, batch_size);
    
    batch_ntt_forward_kernel<<<blocks, threads, 0, stream>>>(
        data_batch, batch_size, n, log_n, mod_idx
    );
    
    return cudaGetLastError();
}

cudaError_t launch_batch_he_add(uint64_t** ct1_c0_batch, uint64_t** ct1_c1_batch,
                                uint64_t** ct2_c0_batch, uint64_t** ct2_c1_batch,
                                uint64_t** res_c0_batch, uint64_t** res_c1_batch,
                                uint32_t batch_size, uint32_t n, uint32_t mod_idx,
                                cudaStream_t stream) {
    uint32_t threads = 256;
    dim3 blocks((n + threads - 1) / threads, batch_size);
    
    batch_he_add_kernel<<<blocks, threads, 0, stream>>>(
        ct1_c0_batch, ct1_c1_batch, ct2_c0_batch, ct2_c1_batch,
        res_c0_batch, res_c1_batch, batch_size, n, mod_idx
    );
    
    return cudaGetLastError();
}

// Memory utilities
cudaError_t launch_zero_memory(uint64_t* data, size_t size, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;
    
    zero_memory_kernel<<<blocks, threads, 0, stream>>>(data, size);
    
    return cudaGetLastError();
}

cudaError_t launch_copy_memory(const uint64_t* src, uint64_t* dst, size_t size,
                              cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;
    
    copy_memory_kernel<<<blocks, threads, 0, stream>>>(src, dst, size);
    
    return cudaGetLastError();
}

// Noise estimation
cudaError_t launch_estimate_noise(const uint64_t* ciphertext, double* noise_estimate,
                                 uint32_t n, uint32_t level, double scale,
                                 cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(double);
    
    estimate_noise_kernel<<<blocks, threads, shared_mem, stream>>>(
        ciphertext, noise_estimate, n, level, scale
    );
    
    return cudaGetLastError();
}

// Stream management
cudaError_t cuda_create_stream(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

cudaError_t cuda_destroy_stream(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

cudaError_t cuda_stream_synchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

} // extern "C"

} // namespace cuda
} // namespace hegraph