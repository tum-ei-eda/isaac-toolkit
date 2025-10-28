# Coremark Standalone Example

## Prerequisites

See [`../SETUP.md`](../SETUP.md) for details!

**Warning:** Make sure that your `$INSTALL_DIR` environment variable contains an **absolute** and valid path, e.g. `export INSTALL_DIR=$(pwd)/../install/`.

*Hint:* Other RISC-V configs (different arch/abi) can also be used when downloading and using the appropriate toolchains.

For the profiling step, additional Python packages should be installed:

```sh
pip install -r ../requirements.txt
```

## NEW: Simplified Usage via Makefile

Examples:

```sh
# End-to-End Flow
make all
make SIMULATOR=etiss all

# Clean Artifacts
make clean

# Override Artifacts
make all FORCE=1

# Single steps
make init
make compile
make run
make load
make analyze
make visualize
make profile
make callgraph

# Extra options
make compile RISCV_ARCH=rv32gc RISCV_ABI=ilp32d
make compile OPTIMIZE=s

```
