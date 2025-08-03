#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <complex>

namespace hegraph {

class CKKSParameters {
public:
    uint32_t poly_modulus_degree;
    std::vector<uint64_t> coeff_modulus_bits;
    double scale;
    uint32_t security_level;
    
    CKKSParameters(uint32_t poly_degree = 32768,
                   const std::vector<uint64_t>& coeff_bits = {60, 40, 40, 40, 40, 60},
                   double scale_val = 1ULL << 40,
                   uint32_t sec_level = 128)
        : poly_modulus_degree(poly_degree),
          coeff_modulus_bits(coeff_bits),
          scale(scale_val),
          security_level(sec_level) {}
    
    bool validate() const;
    uint64_t get_total_coeff_modulus_bits() const;
};

class Ciphertext {
public:
    std::vector<std::vector<uint64_t>> c0;
    std::vector<std::vector<uint64_t>> c1;
    double scale;
    uint32_t level;
    uint32_t size;
    
    Ciphertext() : scale(0), level(0), size(0) {}
    Ciphertext(uint32_t poly_degree, uint32_t num_primes);
    
    void resize(uint32_t poly_degree, uint32_t num_primes);
    bool is_valid() const;
    uint32_t noise_budget() const;
};

class Plaintext {
public:
    std::vector<std::complex<double>> values;
    double scale;
    
    Plaintext() : scale(0) {}
    explicit Plaintext(const std::vector<std::complex<double>>& vals, double s = 0)
        : values(vals), scale(s) {}
    
    size_t slot_count() const { return values.size(); }
};

class SecretKey {
private:
    std::vector<std::vector<int64_t>> poly;
    
public:
    SecretKey() = default;
    explicit SecretKey(uint32_t poly_degree, uint32_t num_primes);
    
    void generate(uint32_t poly_degree, uint32_t num_primes);
    const std::vector<std::vector<int64_t>>& get_poly() const { return poly; }
};

class PublicKey {
public:
    Ciphertext pk0;
    Ciphertext pk1;
    
    PublicKey() = default;
    PublicKey(uint32_t poly_degree, uint32_t num_primes);
    
    void generate(const SecretKey& sk, const CKKSParameters& params);
};

class RelinearizationKey {
public:
    std::vector<std::pair<Ciphertext, Ciphertext>> keys;
    uint32_t decomposition_bit_count;
    
    RelinearizationKey() : decomposition_bit_count(0) {}
    void generate(const SecretKey& sk, const CKKSParameters& params);
};

class GaloisKey {
public:
    std::vector<std::pair<uint32_t, std::pair<Ciphertext, Ciphertext>>> keys;
    
    void generate(const SecretKey& sk, const std::vector<uint32_t>& galois_elts,
                  const CKKSParameters& params);
};

class CKKSContext {
private:
    CKKSParameters params;
    std::unique_ptr<SecretKey> secret_key;
    std::unique_ptr<PublicKey> public_key;
    std::unique_ptr<RelinearizationKey> relin_keys;
    std::unique_ptr<GaloisKey> galois_keys;
    
    // NTT tables for optimization
    std::vector<std::vector<uint64_t>> ntt_tables;
    std::vector<std::vector<uint64_t>> intt_tables;
    
    // GPU context
    int gpu_device_id;
    void* cuda_stream;
    
    void initialize_ntt_tables();
    void setup_gpu_context();
    
public:
    explicit CKKSContext(const CKKSParameters& parameters, int gpu_id = 0);
    ~CKKSContext();
    
    // Key generation
    void generate_keys();
    
    // Encryption/Decryption
    Ciphertext encrypt(const Plaintext& plain) const;
    Plaintext decrypt(const Ciphertext& cipher) const;
    
    // Homomorphic operations
    Ciphertext add(const Ciphertext& a, const Ciphertext& b) const;
    Ciphertext multiply(const Ciphertext& a, const Ciphertext& b) const;
    Ciphertext multiply_plain(const Ciphertext& cipher, const Plaintext& plain) const;
    Ciphertext rotate(const Ciphertext& cipher, int steps) const;
    
    // Noise management
    Ciphertext relinearize(const Ciphertext& cipher) const;
    Ciphertext rescale(const Ciphertext& cipher) const;
    Ciphertext mod_switch(const Ciphertext& cipher) const;
    
    // Advanced operations
    Ciphertext bootstrap(const Ciphertext& cipher) const;
    std::vector<Ciphertext> batch_encrypt(const std::vector<Plaintext>& plains) const;
    
    // Getters
    const CKKSParameters& get_params() const { return params; }
    uint32_t get_noise_budget(const Ciphertext& cipher) const;
    bool validate_ciphertext(const Ciphertext& cipher) const;
    
    // GPU operations
    void to_gpu(Ciphertext& cipher) const;
    void to_cpu(Ciphertext& cipher) const;
    
    // Security validation
    bool validate_security_level() const;
    uint32_t estimate_security_bits() const;
};

// Utility functions
std::vector<std::complex<double>> encode_vector(const std::vector<double>& real_values);
std::vector<double> decode_real(const std::vector<std::complex<double>>& complex_values);

// Error handling
class CKKSException : public std::exception {
private:
    std::string message;
public:
    explicit CKKSException(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

} // namespace hegraph