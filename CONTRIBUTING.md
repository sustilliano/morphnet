# Contributing to MorphNet-GTL

We love contributions! MorphNet-GTL is an ambitious project to revolutionize spatial intelligence, and we welcome contributors from all backgrounds.

## 🚀 Quick Start

1. **Fork** the repository
1. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/morphnet.git`
1. **Create** a branch: `git checkout -b feature/amazing-feature`
1. **Make** your changes
1. **Test** your changes: `cargo test`
1. **Submit** a pull request

## 🛠️ Development Setup

### Prerequisites

- **Rust 1.70+** (stable)
- **Git**
- **A good IDE** (VS Code with rust-analyzer recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/sustilliano/morphnet.git
cd morphnet

# Build the project
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check formatting and linting
cargo fmt --check
cargo clippy -- -D warnings
```

## 🎯 Ways to Contribute

### 🐛 Bug Reports

Found a bug? Please [create an issue](https://github.com/sustilliano/morphnet/issues/new) with:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Rust version, etc.)
- **Minimal code example** if possible

### ✨ Feature Requests

Have an idea? We’d love to hear it! [Open an issue](https://github.com/sustilliano/morphnet/issues/new) describing:

- **The problem** you’re trying to solve
- **Your proposed solution**
- **Alternative approaches** you’ve considered
- **Use cases** and examples

### 🔧 Code Contributions

#### High-Priority Areas

- **🧠 New Body Plans**: Add support for more anatomical structures (fish, insects, custom forms)
- **📊 Performance**: Optimize critical paths and memory usage
- **🔌 Integrations**: Connect with PyTorch, TensorFlow, or other ML frameworks
- **📖 Documentation**: Improve guides, examples, and API docs
- **🧪 Testing**: Add more comprehensive test coverage

#### Code Style

We follow standard Rust conventions:

- **Format code**: `cargo fmt`
- **Lint code**: `cargo clippy`
- **Write tests** for all new functionality
- **Document public APIs** with doc comments
- **Keep functions small** and focused
- **Use descriptive names** for variables and functions

## 📝 Pull Request Process

### Before Submitting

1. **Rebase** your branch on the latest main
1. **Run tests**: `cargo test`
1. **Check formatting**: `cargo fmt --check`
1. **Run clippy**: `cargo clippy -- -D warnings`
1. **Update documentation** if needed
1. **Add tests** for new functionality

### PR Template

When submitting a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added/updated tests
- [ ] All tests pass
- [ ] Benchmarks run successfully

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Breaking changes documented
```

### Review Process

1. **Automated checks** must pass (CI/CD)
1. **Code review** by maintainers
1. **Discussion** and feedback
1. **Approval** and merge

## 🧪 Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_creation() {
        let template = TemplateFactory::create_quadruped();
        assert_eq!(template.body_plan, BodyPlan::Quadruped);
        assert!(!template.keypoints.is_empty());
    }
}
```

### Integration Tests

Place in `tests/` directory:

```rust
use morphnet_gtl::prelude::*;

#[test]
fn test_full_pipeline() {
    // Test complete workflows
}
```

### Benchmarks

Add to `benches/` directory:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_classification(c: &mut Criterion) {
    c.bench_function("classify_image", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });
}

criterion_group!(benches, benchmark_classification);
criterion_main!(benches);
```

## 📚 Documentation

### Code Documentation

- **Public APIs** must have doc comments
- **Examples** in doc comments when helpful
- **Link to related functions** with `[function_name]`

```rust
/// Classifies an image and predicts geometric template.
/// 
/// # Arguments
/// * `image` - Input image as 3D array (H×W×C)
/// 
/// # Returns
/// Classification result with species, body plan, and template
/// 
/// # Example
/// ```
/// use morphnet_gtl::prelude::*;
/// 
/// let morphnet = MorphNetBuilder::new().build()?;
/// let result = morphnet.classify(&image)?;
/// println!("Species: {}", result.predicted_species);
/// ```
pub fn classify(&self, image: &Array3<f32>) -> Result<ClassificationResult> {
    // Implementation
}
```

### User Documentation

- **Getting started guides** in `docs/`
- **API reference** via `cargo doc`
- **Examples** in `examples/` directory
- **README** updates for new features

## 🏗️ Architecture Guidelines

### Design Principles

1. **Modularity**: Each component should be independently testable
1. **Performance**: Optimize for real-time use cases
1. **Extensibility**: Easy to add new body plans and features
1. **Safety**: Leverage Rust’s type system for correctness
1. **Documentation**: Code should be self-documenting

### Module Structure

```
src/
├── lib.rs              # Public API and re-exports
├── mmx/                # Multimedia Matrix format
├── morphnet/           # Neural network and templates
├── patch_quilt/        # Mesh refinement system
├── analysis/           # Analytics and visualization
└── spatial/            # Spatial awareness and prediction
```

### Error Handling

- Use `Result<T, Error>` for fallible operations
- Create specific error types with `thiserror`
- Provide helpful error messages
- Chain errors with context

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Welcome newcomers** and help them learn
- **Focus on constructive** feedback
- **Collaborate** rather than compete
- **Give credit** where it’s due

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

## 📈 Performance Considerations

### Benchmarking

- **Measure before optimizing**
- **Use criterion** for micro-benchmarks
- **Profile with perf** or similar tools
- **Test on real-world data**

### Memory Management

- **Minimize allocations** in hot paths
- **Use object pooling** for frequent allocations
- **Consider zero-copy** operations
- **Profile memory usage** with valgrind/heaptrack

### Concurrency

- **Use rayon** for data parallelism
- **Consider async** for I/O operations
- **Avoid locks** in performance-critical code
- **Test thread safety** thoroughly

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Git tag created
- [ ] Crates.io published

## 💡 Ideas for Contributions

### Beginner-Friendly

- **Add examples** for specific use cases
- **Improve error messages** with better context
- **Write documentation** for existing features
- **Add unit tests** for uncovered code
- **Fix clippy warnings**

### Intermediate

- **Implement new body plans** (fish, insects, etc.)
- **Add visualization tools** for templates
- **Optimize performance** in critical paths
- **Add integration tests** for complex scenarios
- **Improve benchmarking** suite

### Advanced

- **Implement neural network training**
- **Add GPU acceleration** support
- **Create Python bindings** via PyO3
- **Build web assembly** version
- **Add distributed processing** capabilities

## 📞 Questions?

- **General questions**: [GitHub Discussions](https://github.com/sustilliano/morphnet/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/sustilliano/morphnet/issues)
- **Direct contact**: asustello@gmail.com

Thank you for contributing to MorphNet-GTL! Together, we’re building the future of spatial intelligence.
