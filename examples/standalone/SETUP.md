# Setup Instructions

The download/installation instructions for several tools are described below.

It is recommended to install all tools into a directory called `install/` (can be changed via `export INSTALL_DIR=...`).

#### Notes

Utilities in the [`scripts/`](scripts/) directory are used to select the appropriate files to download and unpack them automatically.

Most tools are hosted [here](https://github.com/PhilippvK/riscv-tools/releases) and can be manually downloaded via:

```sh
wget https://github.com/PhilippvK/riscv-tools/releases/download/gnu_2024.09.03/riscv64-unknown-elf-ubuntu-20.04-rv32gc_ilp32d.tar.xz
```

or similar.

## RISC-V Toolchains

### RISC-V GNU Tools (GCC)

```sh
# rv32im_ilp32
./scripts/download_helper.sh $INSTALL_DIR/rv32im_ilp32/ gnu 2024.09.03 rv32im_zicsr_zifencei_ilp32
```

### LLVM

```sh
./scripts/download_helper.sh $INSTALL_DIR/llvm/ llvm 19.1.1
```

## Simulators

### ETISS

```sh
# NOT IMPLEMENTED!
# ./scripts/download_helper.sh $INSTALL_DIR/etiss/ etiss 2024.11.28

# Alternative:
./scripts/setup_etiss.sh $INSTALL_DIR/etiss
```

### Spike

```sh
./scripts/download_helper.sh $INSTALL_DIR/spike/ spike 2024.11.28
```
