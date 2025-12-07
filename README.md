# SpTeCE - Sparse Tensor Computation Engine

[![CI/CD](https://github.com/yourusername/sptece/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/sptece/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/sptece.svg)](https://crates.io/crates/sptece)
[![Documentation](https://docs.rs/sptece/badge.svg)](https://docs.rs/sptece)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Sparse Tensor Computation Engine for Rust** – A high-performance library for sparse tensor operations (like a Rust-native PyTorch sparse or TensorFlow tf.sparse).

## 🚀 Project Status

**Phase 0/6 Complete** – Infrastructure Setup  
**Next Phase**: Core Tensor Abstraction (Month 1)

## 📦 Crate Structure

This is a Cargo workspace with these crates:

| Crate | Purpose | Status |
|-------|---------|--------|
| `sparse-core` | Core traits and types | 🔨 In Progress |
| `sparse-csr` | CSR format implementation | ✅ Ready (from existing code) |
| `sparse-coo` | COO format implementation | 📅 Planned |
| `sparse-ops` | Tensor operations | 📅 Planned |
| `sparse-autograd` | Automatic differentiation | 📅 Planned |
| `sparse-nn` | Neural network layers | 📅 Planned |
| `sparse-io` | Serialization and file I/O | 📅 Planned |

## 🛠️ Development

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Just command runner (`cargo install just`)

### Common Commands

```bash
# Check code quality
just qa

# Run tests
just test

# Build documentation
just doc

# Run benchmarks
just bench
