#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "../ckks/ckks_context.h"
#include <complex>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace hegraph;

// Helper function to convert Python list to std::vector<double>
std::vector<double> py_list_to_double_vector(const py::list& py_list) {
    std::vector<double> result;
    result.reserve(py_list.size());
    for (auto item : py_list) {
        result.push_back(item.cast<double>());
    }
    return result;
}

// Helper function to convert std::vector<double> to Python list
py::list double_vector_to_py_list(const std::vector<double>& vec) {
    py::list result;
    for (const auto& val : vec) {
        result.append(val);
    }
    return result;
}

// Helper function to convert Python list to std::vector<std::complex<double>>
std::vector<std::complex<double>> py_list_to_complex_vector(const py::list& py_list) {
    std::vector<std::complex<double>> result;
    result.reserve(py_list.size());
    for (auto item : py_list) {
        if (py::isinstance<py::float_>(item) || py::isinstance<py::int_>(item)) {
            result.emplace_back(item.cast<double>(), 0.0);
        } else {
            result.push_back(item.cast<std::complex<double>>());
        }
    }
    return result;
}

// Helper function to convert std::vector<std::complex<double>> to Python list
py::list complex_vector_to_py_list(const std::vector<std::complex<double>>& vec) {
    py::list result;
    for (const auto& val : vec) {
        result.append(val);
    }
    return result;
}

PYBIND11_MODULE(_hegraph_bindings, m) {
    m.doc() = "HE-Graph-Embeddings: Python bindings for CKKS homomorphic encryption";

    // Exception class
    py::register_exception<CKKSException>(m, "CKKSException");

    // CKKSParameters class
    py::class_<CKKSParameters>(m, "CKKSParameters")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, const std::vector<uint64_t>&, double, uint32_t>(),
             "Constructor with parameters",
             py::arg("poly_modulus_degree") = 32768,
             py::arg("coeff_modulus_bits") = std::vector<uint64_t>{60, 40, 40, 40, 40, 60},
             py::arg("scale") = 1ULL << 40,
             py::arg("security_level") = 128)
        .def_readwrite("poly_modulus_degree", &CKKSParameters::poly_modulus_degree)
        .def_readwrite("coeff_modulus_bits", &CKKSParameters::coeff_modulus_bits)
        .def_readwrite("scale", &CKKSParameters::scale)
        .def_readwrite("security_level", &CKKSParameters::security_level)
        .def("validate", &CKKSParameters::validate, "Validate parameter configuration")
        .def("get_total_coeff_modulus_bits", &CKKSParameters::get_total_coeff_modulus_bits,
             "Get total coefficient modulus bits")
        .def("__repr__", [](const CKKSParameters& params) {
            return "<CKKSParameters poly_degree=" + std::to_string(params.poly_modulus_degree) +
                   " security_level=" + std::to_string(params.security_level) + ">";
        });

    // Plaintext class
    py::class_<Plaintext>(m, "Plaintext")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<std::complex<double>>&, double>(),
             "Constructor with values and scale",
             py::arg("values"), py::arg("scale") = 0.0)
        .def_readwrite("values", &Plaintext::values)
        .def_readwrite("scale", &Plaintext::scale)
        .def("slot_count", &Plaintext::slot_count, "Get number of slots")
        .def("__len__", &Plaintext::slot_count)
        .def("__repr__", [](const Plaintext& plain) {
            return "<Plaintext slots=" + std::to_string(plain.slot_count()) +
                   " scale=" + std::to_string(plain.scale) + ">";
        })
        // Python-friendly interface
        .def("from_real_list", [](Plaintext& plain, const py::list& values, double scale) {
            std::vector<double> real_vals = py_list_to_double_vector(values);
            plain.values = encode_vector(real_vals);
            plain.scale = scale;
        }, "Create plaintext from real values list")
        .def("from_complex_list", [](Plaintext& plain, const py::list& values, double scale) {
            plain.values = py_list_to_complex_vector(values);
            plain.scale = scale;
        }, "Create plaintext from complex values list")
        .def("to_real_list", [](const Plaintext& plain) {
            std::vector<double> real_vals = decode_real(plain.values);
            return double_vector_to_py_list(real_vals);
        }, "Convert plaintext to real values list")
        .def("to_complex_list", [](const Plaintext& plain) {
            return complex_vector_to_py_list(plain.values);
        }, "Convert plaintext to complex values list");

    // Ciphertext class (also serves as EncryptedTensor)
    py::class_<Ciphertext>(m, "Ciphertext")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, uint32_t>(), "Constructor with polynomial degree and prime count",
             py::arg("poly_degree"), py::arg("num_primes"))
        .def_readwrite("scale", &Ciphertext::scale)
        .def_readwrite("level", &Ciphertext::level)
        .def_readonly("size", &Ciphertext::size)
        .def("resize", &Ciphertext::resize, "Resize ciphertext",
             py::arg("poly_degree"), py::arg("num_primes"))
        .def("is_valid", &Ciphertext::is_valid, "Check if ciphertext is valid")
        .def("noise_budget", &Ciphertext::noise_budget, "Get remaining noise budget")
        .def("__repr__", [](const Ciphertext& cipher) {
            return "<Ciphertext level=" + std::to_string(cipher.level) +
                   " scale=" + std::to_string(cipher.scale) +
                   " size=" + std::to_string(cipher.size) + ">";
        })
        // EncryptedTensor-like properties
        .def_property("c0", 
            [](const Ciphertext& cipher) { 
                return py::cast(cipher.c0);
            },
            [](Ciphertext& cipher, const py::object& val) {
                // Setter for c0 - simplified for bindings
            })
        .def_property("c1", 
            [](const Ciphertext& cipher) { 
                return py::cast(cipher.c1);
            },
            [](Ciphertext& cipher, const py::object& val) {
                // Setter for c1 - simplified for bindings
            })
        // Operator overloads for Python-style operations
        .def("__add__", [](const Ciphertext& a, const Ciphertext& b) {
            throw std::runtime_error("Direct addition not supported - use context.add()");
        })
        .def("__mul__", [](const Ciphertext& a, const Ciphertext& b) {
            throw std::runtime_error("Direct multiplication not supported - use context.multiply()");
        })
        .def("rotate", [](const Ciphertext& cipher, int steps) {
            throw std::runtime_error("Direct rotation not supported - use context.rotate()");
        })
        .def("rescale", [](const Ciphertext& cipher) {
            throw std::runtime_error("Direct rescale not supported - use context.rescale()");
        });

    // Create alias for EncryptedTensor
    m.attr("EncryptedTensor") = m.attr("Ciphertext");

    // SecretKey class
    py::class_<SecretKey>(m, "SecretKey")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, uint32_t>(), "Constructor with polynomial degree and prime count",
             py::arg("poly_degree"), py::arg("num_primes"))
        .def("generate", &SecretKey::generate, "Generate secret key",
             py::arg("poly_degree"), py::arg("num_primes"));

    // PublicKey class
    py::class_<PublicKey>(m, "PublicKey")
        .def(py::init<>(), "Default constructor")
        .def(py::init<uint32_t, uint32_t>(), "Constructor with polynomial degree and prime count",
             py::arg("poly_degree"), py::arg("num_primes"))
        .def("generate", &PublicKey::generate, "Generate public key from secret key",
             py::arg("secret_key"), py::arg("params"));

    // RelinearizationKey class
    py::class_<RelinearizationKey>(m, "RelinearizationKey")
        .def(py::init<>(), "Default constructor")
        .def_readonly("decomposition_bit_count", &RelinearizationKey::decomposition_bit_count)
        .def("generate", &RelinearizationKey::generate, "Generate relinearization keys",
             py::arg("secret_key"), py::arg("params"));

    // GaloisKey class
    py::class_<GaloisKey>(m, "GaloisKey")
        .def(py::init<>(), "Default constructor")
        .def("generate", &GaloisKey::generate, "Generate Galois keys",
             py::arg("secret_key"), py::arg("galois_elements"), py::arg("params"));

    // Main CKKSContext class
    py::class_<CKKSContext>(m, "CKKSContext")
        .def(py::init<const CKKSParameters&, int>(), "Constructor",
             py::arg("parameters"), py::arg("gpu_id") = 0)
        .def("generate_keys", &CKKSContext::generate_keys, "Generate all encryption keys")
        
        // Encryption/Decryption
        .def("encrypt", &CKKSContext::encrypt, "Encrypt plaintext",
             py::arg("plaintext"))
        .def("decrypt", &CKKSContext::decrypt, "Decrypt ciphertext",
             py::arg("ciphertext"))
        
        // Homomorphic operations
        .def("add", &CKKSContext::add, "Homomorphic addition",
             py::arg("a"), py::arg("b"))
        .def("multiply", &CKKSContext::multiply, "Homomorphic multiplication",
             py::arg("a"), py::arg("b"))
        .def("multiply_plain", &CKKSContext::multiply_plain, "Multiply ciphertext with plaintext",
             py::arg("ciphertext"), py::arg("plaintext"))
        .def("rotate", &CKKSContext::rotate, "Rotate encrypted vector",
             py::arg("ciphertext"), py::arg("steps"))
        
        // Noise management
        .def("relinearize", &CKKSContext::relinearize, "Relinearize ciphertext",
             py::arg("ciphertext"))
        .def("rescale", &CKKSContext::rescale, "Rescale ciphertext",
             py::arg("ciphertext"))
        .def("mod_switch", &CKKSContext::mod_switch, "Modulus switching",
             py::arg("ciphertext"))
        
        // Advanced operations
        .def("bootstrap", &CKKSContext::bootstrap, "Bootstrap ciphertext",
             py::arg("ciphertext"))
        .def("batch_encrypt", &CKKSContext::batch_encrypt, "Batch encrypt multiple plaintexts",
             py::arg("plaintexts"))
        
        // Getters and validation
        .def("get_params", &CKKSContext::get_params, "Get CKKS parameters",
             py::return_value_policy::reference)
        .def("get_noise_budget", &CKKSContext::get_noise_budget, "Get noise budget of ciphertext",
             py::arg("ciphertext"))
        .def("validate_ciphertext", &CKKSContext::validate_ciphertext, "Validate ciphertext",
             py::arg("ciphertext"))
        
        // GPU operations
        .def("to_gpu", &CKKSContext::to_gpu, "Move ciphertext to GPU",
             py::arg("ciphertext"))
        .def("to_cpu", &CKKSContext::to_cpu, "Move ciphertext to CPU",
             py::arg("ciphertext"))
        
        // Security
        .def("validate_security_level", &CKKSContext::validate_security_level,
             "Validate security level")
        .def("estimate_security_bits", &CKKSContext::estimate_security_bits,
             "Estimate security bits")
        
        .def("__repr__", [](const CKKSContext& ctx) {
            return "<CKKSContext poly_degree=" + 
                   std::to_string(ctx.get_params().poly_modulus_degree) +
                   " security_level=" + 
                   std::to_string(ctx.get_params().security_level) + ">";
        })
        
        // Python-friendly factory method
        .def_static("from_config", [](const py::dict& config) {
            CKKSParameters params;
            
            if (config.contains("poly_modulus_degree")) {
                params.poly_modulus_degree = config["poly_modulus_degree"].cast<uint32_t>();
            }
            
            if (config.contains("coeff_modulus_bits")) {
                params.coeff_modulus_bits = config["coeff_modulus_bits"].cast<std::vector<uint64_t>>();
            }
            
            if (config.contains("scale")) {
                params.scale = config["scale"].cast<double>();
            }
            
            if (config.contains("security_level")) {
                params.security_level = config["security_level"].cast<uint32_t>();
            }
            
            int gpu_id = 0;
            if (config.contains("gpu_id")) {
                gpu_id = config["gpu_id"].cast<int>();
            }
            
            return std::make_unique<CKKSContext>(params, gpu_id);
        }, "Create CKKSContext from dictionary configuration")
        
        // Python-friendly encrypt/decrypt methods
        .def("encrypt_real_list", [](CKKSContext& ctx, const py::list& values, double scale) {
            std::vector<double> real_vals = py_list_to_double_vector(values);
            std::vector<std::complex<double>> complex_vals = encode_vector(real_vals);
            Plaintext plain(complex_vals, scale);
            return ctx.encrypt(plain);
        }, "Encrypt list of real values", py::arg("values"), py::arg("scale"))
        
        .def("encrypt_complex_list", [](CKKSContext& ctx, const py::list& values, double scale) {
            std::vector<std::complex<double>> complex_vals = py_list_to_complex_vector(values);
            Plaintext plain(complex_vals, scale);
            return ctx.encrypt(plain);
        }, "Encrypt list of complex values", py::arg("values"), py::arg("scale"))
        
        .def("decrypt_to_real_list", [](CKKSContext& ctx, const Ciphertext& cipher) {
            Plaintext plain = ctx.decrypt(cipher);
            std::vector<double> real_vals = decode_real(plain.values);
            return double_vector_to_py_list(real_vals);
        }, "Decrypt to list of real values", py::arg("ciphertext"))
        
        .def("decrypt_to_complex_list", [](CKKSContext& ctx, const Ciphertext& cipher) {
            Plaintext plain = ctx.decrypt(cipher);
            return complex_vector_to_py_list(plain.values);
        }, "Decrypt to list of complex values", py::arg("ciphertext"));

    // Utility functions
    m.def("encode_vector", &encode_vector, "Encode real vector to complex",
          py::arg("real_values"));
    m.def("decode_real", &decode_real, "Decode complex vector to real",
          py::arg("complex_values"));

    // Constants and enums
    m.attr("DEFAULT_POLY_MODULUS_DEGREE") = 32768;
    m.attr("DEFAULT_SCALE") = 1ULL << 40;
    m.attr("DEFAULT_SECURITY_LEVEL") = 128;
    
    // Security levels (as constants since enum not in C++ header)
    m.attr("SECURITY_128") = 128;
    m.attr("SECURITY_192") = 192;
    m.attr("SECURITY_256") = 256;

    // Version information
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "HE-Graph-Embeddings Team";
}