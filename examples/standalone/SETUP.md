# Setup Instructions

The download/installation instructions for several tools are described below.

**Hint:** It is recommended to install all tools into a directory called `install/` (can be changed via `export INSTALL_DIR=...`).

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
./scripts/download_helper.sh $INSTALL_DIR/rv32im_ilp32/ gnu 2025.06.13 rv32im_zicsr_zifencei_ilp32

# rv32im_zve32x_ilp32
./scripts/download_helper.sh $INSTALL_DIR/rv32im_zve32x_ilp32/ gnu 2025.06.13 rv32im_zicsr_zifencei_zve32x_ilp32

# rv32gc_ilp32d
./scripts/download_helper.sh $INSTALL_DIR/rv32gc_ilp32d/ gnu 2025.06.13 rv32gc_ilp32d

# rv32gcv_ilp32d
./scripts/download_helper.sh $INSTALL_DIR/rv32gcv_ilp32d/ gnu 2025.06.13 rv32gcv_ilp32d

# rv64gc_lp64d
./scripts/download_helper.sh $INSTALL_DIR/rv64gc_lp64d/ gnu 2025.06.13 rv64gc_lp64d riscv64-unknown-elf

# rv64gcv_lp64d
./scripts/download_helper.sh $INSTALL_DIR/rv64gcv_lp64d/ gnu 2025.06.13 rv64gcv_lp64d riscv64-unknown-elf
```

### LLVM

```sh
# llvm 19
./scripts/download_helper.sh $INSTALL_DIR/llvm/ llvm 19.1.1

# llvm 20
./scripts/download_helper.sh $INSTALL_DIR/llvm/ llvm 0.1.8
```

## Simulators

### ETISS

```sh
# prebuilt
# ./scripts/download_helper.sh $INSTALL_DIR/etiss/install etiss 2024.11.28
# ./scripts/setup_etiss_examples.sh $INSTALL_DIR/etiss
# 
# from source:
./scripts/setup_etiss.sh $INSTALL_DIR/etiss
./scripts/setup_etiss_examples.sh $INSTALL_DIR/etiss
```

### Spike

Make sure to have `device-tree-compiler` (`dtc`) installed on your system.

```sh
./scripts/download_helper.sh $INSTALL_DIR/spike/ spike 0bc176b3
```

