# Contributing to SpTeCE

First off, thank you for considering contributing to **SpTeCE**! It's people like you that make SpTeCE such a great project. We appreciate all forms of contributions, from reporting bugs to suggesting new features or writing code.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainer.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please help us by reporting it! Before creating bug reports, please check the **existing issues** as you might find out that the issue has already been reported or resolved.

**How to Submit a Good Bug Report:**

1. **Use a clear and descriptive title** for the issue to identify the problem.
2. **Describe the exact steps to reproduce** the behavior. Provide a minimal, reproducible example if possible, like:

    ```bash
    cargo run --example bug_reproduction
    ```

3. **Include specific examples** (code snippets, error messages, full stack traces) that demonstrate the issue.
4. **Describe the expected behavior** and the **actual behavior** you observed.
5. **Include details about your environment**, especially when reporting a build or execution issue:

    ```bash
    rustc --version
    cargo --version
    uname -a # or systeminfo on Windows
    ```

### Suggesting Enhancements

Suggestions for new features, major refactoring ideas, or improvements to existing functionality are welcome! Before suggesting enhancements, please check the **existing issues** to see if your idea has already been proposed or is currently being worked on.

**How to Suggest a Good Enhancement:**

1. **Use a clear and descriptive title.**
2. **Provide a step-by-step description** of the suggested enhancement.
3. **Explain why this enhancement would be useful** to SpTeCE users.
4. **Include examples** of how the new feature would be used (e.g., sample code, configuration examples).

## Code Contributions (Pull Requests)

The following steps outline how to contribute code changes to SpTeCE.

### Setup and Prerequisites

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

    ```bash
    git clone [https://github.com/YOUR_USERNAME/SpTeCE.git](https://github.com/YOUR_USERNAME/SpTeCE.git)
    cd SpTeCE
    ```

3. **Install Rust:** Ensure you have a recent, stable version of **Rust** installed.
4. **Verify the build:**

    ```bash
    cargo test
    ```

### Making Changes

1. **Create a new branch** for your work. Use a descriptive name like `fix/issue-123-bug-name` or `feat/new-feature-name`:

    ```bash
    git checkout -b branch-name
    ```

2. **Make your changes.** Ensure your code follows the existing style, runs the existing tests, and passes any linting checks.
3. **Write tests** for your changes! Bug fixes should include a test that fails without the fix and passes with it. New features should be covered by appropriate unit or integration tests.
4. **Commit your changes** using a clear, descriptive commit message. If your work addresses an issue, reference it in the message (e.g., `git commit -m "Fix: Issue #123 - Prevent crash on empty input."`).

### Submitting the Pull Request

1. **Push your branch** to your fork on GitHub:

    ```bash
    git push origin branch-name
    ```

2. **Open a Pull Request (PR)** on the main SpTeCE repository.
3. In the PR description, include:
    * A summary of the changes.
    * Any relevant issue numbers (e.g., `Closes #123`).
    * Proof that tests are passing.

The maintainers will review your PR, offer feedback, and merge it when it's ready.

## Development Setup

**Prerequisites**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
cargo install just      # Task runner
cargo install cargo-edit # Dependency management
```

**Cloning the Repository**

```bash
git clone forked_repo_url
cd SpTeCE
```

**Building and Testing**

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run quality checks
just qa

# Run tests
just test

# Run benchmarks
just bench
```

## Coding Standards

We follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) and [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/) to maintain code quality and consistency.

1. Naming conventions:
   * Structs, traits, enums: PascalCase
   * Functions, variables: snake_case
   * Constants: SCREAMING_SNAKE_CASE

2. Documentation:

```rust
/// Brief summary on first line
///
/// Detailed description explaining the purpose,
/// any invariants, and examples.
///
/// # Examples
/// ```
/// use sptece::Tensor;
/// let tensor = Tensor::new(&[2, 3]);
/// ```
///
/// # Errors
/// Returns `Error::InvalidShape` if...
///
/// # Panics
/// Panics if...
pub fn meaningful_name(&self) -> Result<(), Error> {
    // Implementation
}
```

3. Error Handling:
   * Use `Result<T, E>` for recoverable errors.
   * Use `panic!` only for unrecoverable errors.
   * Use`thiserror` crate for error types.
   * Provide clear error messages and context.
   * Never use `unwrap()` or `expect()` in library code.

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages. A typical commit message looks like this:

```text
feat: add sparse tensor contraction
fix(csr): handle empty matrix in transpose
docs: update README with benchmarks
test: add property tests for COO format
refactor: simplify tensor trait hierarchy
perf: optimize SpMM kernel by 30%
```

## Testing Requirements

1.

**Unit tests** go in the same module.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        // Test code
    }
}
```

2. **Integration tests** go in `tests/` directory.

3. **Property-based tests** using `proptest` for complex logic.
4. **Benchmarks** using `criterion` for performance-critical code.

## Project Structure

```
crates/
├── sparse-core/     # Core traits and types
├── sparse-csr/      # CSR format implementation
├── sparse-coo/      # COO format implementation
├── sparse-ops/      # Tensor operations
├── sparse-autograd/ # Automatic differentiation
├── sparse-nn/       # Neural network layers
└── sparse-io/       # Serialization and file I/O
```

## Adding a New Crate

1. Create directory in `crates/`
2. Add `Cargo.toml` with proper dependencies
3. Add crate to workspace root `Cargo.toml`
4. Create minimal working example
5. Add comprehensive tests

Performance Considerations

* **Benchmark changes** that affect hot paths
* **Use** `#[inline]` judiciously for small, frequently called functions
* **Avoid allocations** in tight loops
* **Profile** before optimizing with `cargo flamegraph`

## Documentation

1. Public APIs must be documented.
2. Examples should be included in doc comments.
3. Update CHANGELOG.md for user-facing changes.
4. Keep `README.md` up to date with installation and usage instructions.

## Release Process

1. **All test pass** (`just test`).
2. **Benchmarks show no regressions** (`just bench`).
3. **Documentation** is updated.
4. **CHANGELOG.md** updated with all changes.
5. **Version bumped** following semver.
6. **Taged release** created on GitHub.

## Recognition

Contributors will be:

1. Listed in CONTRIBUTORS.md
2. Mentioned in release notes
3. Eligible for maintainer role after significant contributions

Thank you for contributing to making SpTeCE better! 🚀
