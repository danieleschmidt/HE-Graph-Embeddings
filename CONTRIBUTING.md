# Contributing to HE-Graph-Embeddings

Thank you for your interest in contributing to HE-Graph-Embeddings! This document provides guidelines for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/HE-Graph-Embeddings/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System configuration (GPU model, CUDA version, Python version)
   - Error messages and stack traces

### Suggesting Features

1. Check existing [feature requests](https://github.com/yourusername/HE-Graph-Embeddings/issues?q=is%3Aissue+label%3Aenhancement)
2. Open a discussion in [GitHub Discussions](https://github.com/yourusername/HE-Graph-Embeddings/discussions)
3. Provide:
   - Use case and motivation
   - Proposed implementation approach
   - Potential impact on existing features

### Code Contributions

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/HE-Graph-Embeddings.git
cd HE-Graph-Embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Build CUDA kernels
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
```

#### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following our coding standards
4. Write tests for new functionality
5. Run tests: `pytest tests/`
6. Check code quality: `make lint`
7. Commit with descriptive messages
8. Push to your fork
9. Create a Pull Request

#### Coding Standards

##### C++/CUDA Code
- Follow Google C++ Style Guide
- Use clang-format with provided `.clang-format`
- Document CUDA kernels thoroughly
- Include performance benchmarks for optimizations

##### Python Code
- Follow PEP 8 style guide
- Use type hints for all public APIs
- Maximum line length: 100 characters
- Use Black formatter: `black src/`
- Sort imports with isort: `isort src/`

##### Documentation
- Document all public APIs
- Include docstrings with examples
- Update relevant documentation
- Add ADR for architectural decisions

#### Testing Requirements

- Minimum 90% test coverage for new code
- Include unit and integration tests
- Test both encrypted and plaintext paths
- Verify numerical accuracy within tolerance
- Add security tests for cryptographic code

#### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Example:
```
feat(ckks): add bootstrapping support for deep circuits

Implement automatic bootstrapping when noise budget falls below threshold.
Includes optimized key switching and modulus reduction.

Closes #123
```

### CUDA Kernel Contributions

Special guidelines for GPU optimization:

1. **Performance Profiling**
   - Profile with Nsight Compute
   - Include before/after benchmarks
   - Document memory access patterns

2. **Memory Management**
   - Minimize global memory access
   - Use shared memory effectively
   - Coalesce memory transactions

3. **Documentation**
   - Explain algorithmic approach
   - Document thread/block configuration
   - Include complexity analysis

### Security Contributions

For security-related contributions:

1. **Private Disclosure**
   - Report vulnerabilities to security@example.com
   - Do NOT create public issues
   - Allow 90 days for fixes

2. **Cryptographic Changes**
   - Require review from 2+ cryptography experts
   - Include security proofs or references
   - Update security documentation

3. **Testing**
   - Add comprehensive security tests
   - Verify against known test vectors
   - Include edge cases and attack scenarios

## Review Process

### Pull Request Review

1. **Automated Checks**
   - CI/CD pipeline must pass
   - Code coverage maintained
   - No security vulnerabilities

2. **Code Review**
   - At least 1 maintainer approval
   - 2 approvals for cryptographic changes
   - Address all review comments

3. **Documentation Review**
   - API documentation updated
   - Examples included
   - CHANGELOG.md updated

### Review Timeline

- Initial review: Within 5 business days
- Follow-up reviews: Within 3 business days
- Security issues: Within 24 hours

## Development Resources

### Architecture Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design overview
- [docs/adr/](docs/adr/) - Architecture decision records

### API Documentation
- [Python API Reference](https://he-graph-embeddings.readthedocs.io)
- [CUDA Kernel Documentation](docs/cuda-kernels.md)

### Testing Resources
- [Testing Guide](docs/testing-guide.md)
- [Benchmark Suite](benchmarks/README.md)

### Community
- [GitHub Discussions](https://github.com/yourusername/HE-Graph-Embeddings/discussions)
- [Discord Server](https://discord.gg/he-graph-embeddings)
- [Weekly Developer Calls](docs/meetings.md)

## Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- Academic publications (significant contributions)

## Questions?

- Open a [Discussion](https://github.com/yourusername/HE-Graph-Embeddings/discussions)
- Join our [Discord](https://discord.gg/he-graph-embeddings)
- Email: maintainers@example.com

Thank you for contributing to privacy-preserving machine learning!