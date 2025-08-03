#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

namespace hegraph {
namespace cuda {

// Constants for NTT
__constant__ uint64_t d_modulus[32];  // Prime moduli
__constant__ uint64_t d_roots[32];    // Primitive roots
__constant__ uint64_t d_inv_roots[32]; // Inverse roots

// Modular arithmetic functions
__device__ inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    // Barrett reduction for fast modular multiplication
    uint128_t product = static_cast<uint128_t>(a) * b;
    return product % mod;
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

// Forward NTT kernel
__global__ void ntt_forward_kernel(uint64_t* data, uint32_t n, uint32_t log_n, 
                                  uint32_t mod_idx) {
    extern __shared__ uint64_t shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t idx = bid * blockDim.x + tid;
    
    if (idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    uint64_t root = d_roots[mod_idx];
    
    // Load data to shared memory
    shared_data[tid] = data[idx];
    __syncthreads();
    
    // Cooley-Tukey NTT
    for (uint32_t s = 1; s <= log_n; ++s) {
        uint32_t m = 1 << s;
        uint32_t half_m = m >> 1;
        
        uint64_t w_m = mod_pow(root, (n / m), mod);
        
        for (uint32_t k = tid; k < n; k += blockDim.x) {
            uint32_t j = k & (half_m - 1);
            uint32_t i = ((k >> (s - 1)) << s) | j;
            
            if (i < n && i + half_m < n) {
                uint64_t w = mod_pow(w_m, j, mod);
                butterfly(shared_data[i], shared_data[i + half_m], w, mod);
            }
        }
        __syncthreads();
    }
    
    // Write back to global memory
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
}

// Inverse NTT kernel
__global__ void ntt_inverse_kernel(uint64_t* data, uint32_t n, uint32_t log_n,
                                  uint32_t mod_idx) {
    extern __shared__ uint64_t shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t idx = bid * blockDim.x + tid;
    
    if (idx >= n) return;
    
    uint64_t mod = d_modulus[mod_idx];
    uint64_t inv_root = d_inv_roots[mod_idx];
    
    // Load data to shared memory
    shared_data[tid] = data[idx];
    __syncthreads();
    
    // Gentleman-Sande INTT
    for (uint32_t s = log_n; s >= 1; --s) {
        uint32_t m = 1 << s;
        uint32_t half_m = m >> 1;
        
        uint64_t w_m = mod_pow(inv_root, (n / m), mod);
        
        for (uint32_t k = tid; k < n; k += blockDim.x) {
            uint32_t j = k & (half_m - 1);
            uint32_t i = ((k >> (s - 1)) << s) | j;
            
            if (i < n && i + half_m < n) {
                uint64_t w = mod_pow(w_m, j, mod);
                butterfly(shared_data[i], shared_data[i + half_m], w, mod);
            }
        }
        __syncthreads();
    }
    
    // Scale by n^(-1)
    uint64_t n_inv = mod_pow(n, mod - 2, mod);
    shared_data[tid] = mod_mul(shared_data[tid], n_inv, mod);
    __syncthreads();
    
    // Write back to global memory
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
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

// Rescaling kernel
__global__ void rescale_kernel(uint64_t* data, uint32_t n, uint32_t old_level,
                              uint32_t new_level, double* scale) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Drop the last modulus and adjust scale
    uint64_t dropped_mod = d_modulus[old_level];
    *scale /= static_cast<double>(dropped_mod);
    
    // Data already in correct RNS representation for remaining moduli
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

// Utility kernels

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

void launch_ntt_forward(uint64_t* data, uint32_t n, uint32_t mod_idx,
                       cudaStream_t stream) {
    uint32_t log_n = __builtin_ctz(n);
    uint32_t threads = min(n, 256u);
    uint32_t blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(uint64_t);
    
    ntt_forward_kernel<<<blocks, threads, shared_mem, stream>>>(
        data, n, log_n, mod_idx
    );
}

void launch_ntt_inverse(uint64_t* data, uint32_t n, uint32_t mod_idx,
                       cudaStream_t stream) {
    uint32_t log_n = __builtin_ctz(n);
    uint32_t threads = min(n, 256u);
    uint32_t blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(uint64_t);
    
    ntt_inverse_kernel<<<blocks, threads, shared_mem, stream>>>(
        data, n, log_n, mod_idx
    );
}

void launch_he_add(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                  const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                  uint64_t* res_c0, uint64_t* res_c1,
                  uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    he_add_kernel<<<blocks, threads, 0, stream>>>(
        ct1_c0, ct1_c1, ct2_c0, ct2_c1, res_c0, res_c1, n, mod_idx
    );
}

void launch_he_mul(const uint64_t* ct1_c0, const uint64_t* ct1_c1,
                  const uint64_t* ct2_c0, const uint64_t* ct2_c1,
                  uint64_t* res_c0, uint64_t* res_c1, uint64_t* res_c2,
                  uint32_t n, uint32_t mod_idx, cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    he_mul_kernel<<<blocks, threads, 0, stream>>>(
        ct1_c0, ct1_c1, ct2_c0, ct2_c1, res_c0, res_c1, res_c2, n, mod_idx
    );
}

void launch_rotate(const uint64_t* input, uint64_t* output,
                  uint32_t n, int32_t steps, uint32_t mod_idx,
                  cudaStream_t stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    
    rotate_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n, steps, mod_idx
    );
}

} // extern "C"

} // namespace cuda
} // namespace hegraph