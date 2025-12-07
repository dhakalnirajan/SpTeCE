# SpTeCE Development Commands
# Usage: just <command>

# --- Development ---
default:
    just check

check:
    cargo check --all-features --all-targets

test:
    cargo test --all-features

test-all:
    cargo test --all-features -- --nocapture

test-core:
    cargo test -p sparse-core

test-csr:
    cargo test -p sparse-csr

bench:
    cargo bench --all-features

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-features --all-targets -- -D warnings

doc:
    cargo doc --all-features --no-deps --open

doc-internal:
    RUSTDOCFLAGS="--document-private-items" cargo doc --all-features --no-deps --open

# --- Building ---
build:
    cargo build --release

build-dev:
    cargo build

clean:
    cargo clean

# --- Quality Assurance ---
qa: fmt clippy test
    echo "✅ All quality checks passed"

pre-commit: fmt clippy
    echo "✅ Ready to commit"

# --- Examples ---
run-example:
    cargo run --example {{example}}

# --- Profiling ---
profile:
    cargo build --release && perf record -g ./target/release/examples/{{example}} && perf report

flamegraph:
    cargo flamegraph --example {{example}}

# --- Maintenance ---
update:
    cargo update
    cargo outdated -R

audit:
    cargo audit

# --- Release ---
release-dry-run:
    cargo publish --dry-run -p sparse-core
    cargo publish --dry-run -p sparse-csr
    cargo publish --dry-run -p sparse-coo

release-patch:
    cargo release patch --execute

release-minor:
    cargo release minor --execute

release-major:
    cargo release major --execute

# --- Documentation ---
book:
    mdbook serve docs/book --open

bench-report:
    cargo bench --bench {{bench}} -- --save-baseline {{name}}