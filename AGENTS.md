# Repository Guidelines for LLM Agents

These instructions ensure automated contributions follow project conventions.

## Required checks

Before committing changes, run the following commands from the repository root:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

If any command fails because of environment limitations (for example missing dependencies or network access), note that in the PR description.

## Pull Request Notes

When creating a pull request, include two sections in the body:

- **Summary** – short description of the changes made.
- **Testing** – commands executed and their results.

