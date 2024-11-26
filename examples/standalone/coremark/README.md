# Coremark Standalone Example

## Prerequisites

See [`../SETUP.md`](../SETUP.md) for details!

*Hint:* Other RISC-V configs (different arch/abi) can also be used when downloading and using the appropriate toolchains.

Define configuartion:

```sh
# For isaac toolkit
export SESS=$(pwd)/sess

# For compilation
export BUILD_DIR=$(pwd)/build
export RISCV_PREFIX=$INSTALL_DIR/rv32im_ilp32
export RISCV_NAME=riscv32-unknown-elf-gcc
export RISCV_ARCH=rv32im_zicsr_zifencei
export RISCV_ABI=ilp32
export SYSROOT=$RISCV_PREFIX/$RISCV_NAME
export CC=$RISCV_PREFIX/bin/$RISCV_NAME-gcc
export OBJDUMP=$RISCV_PREFIX/bin/$RISCV_NAME-objdump

# For simulation
# TODO
```

Setup empty ISAAC session:

```sh
python3 -m isaac_toolkit.session.create --session $SESS
```

## Usage

### 1. Compilation

#### Via CMake

TODO
```sh
cmake -S . -B $BUILD_DIR TODO
cmake --build $BUILD_DIR
```

#### Via Makefile

TODO
```sh
make
```

#### Custom (manual)

##### GCC

```sh
mkdir -p $BUILD_DIR
$CC -march=$RISCV_ARCH -mabi=$RISCV_ABI src/*.c -o $BUILD_DIR/coremark.elf -Iinc/ -DITERATIONS=100 -DFLAGS_STR='"testing"' -DPERFORMANCE_RUN -DHAS_STDIO -g -O3 -Xlinker -Map=build/coremark.map
$OBJDUMP -d $BUILD_DIR/coremark.elf > $BUILD_DIR/coremark.dump
```

##### LLVM

TODO

### 2. Static Analysis

Load compilation artifacts into ISAAC session:

```sh
python3 -m isaac_toolkit.frontend.elf.riscv --session $SESS $BUILD_DIR/coremark.elf

# Optional:
# python3 -m isaac_toolkit.frontend.linker_map --session $SESS $BUILD_DIR/coremark.map
# python3 -m isaac_toolkit.frontend.disass.objdump --session $SESS $BUILD_DIR/coremark.dump
```

Run analysis steps:

```sh
python3 -m isaac_toolkit.analysis.static.dwarf --session $SESS
python3 -m isaac_toolkit.analysis.static.mem_footprint --session $SESS

# Optional:
# python3 -m isaac_toolkit.analysis.static.linker_map --session $SESS
# python3 -m isaac_toolkit.analysis.static.histogram.disass_instr --session $SESS
# python3 -m isaac_toolkit.analysis.static.histogram.disass_opcode --session $SESS
```

Generate visualizations:

```sh
python3 -m isaac_toolkit.visualize.pie.mem_footprint --session $SESS --legend

# Optional:
# python3 -m isaac_toolkit.visualize.pie.disass_counts --session $SESS --legend
```

Investigate generated tables:

```sh
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/file2funcs.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/func2pc.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/pc2locs.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/mem_footprint.pkl | less

# Optional:
# python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/symbol_map.pkl | less
# ...
```

Investigate pie charts:

```sh
xdg-open $SESS/plots/mem_footprint_per_func.jpg

# Optional:
# xdg-open $SESS/plots/disass_counts_per_instr.jpg
# xdg-open $SESS/plots/disass_counts_per_opcode.jpg
```

### 3. Simulation

TODO

### 4. Dynamic Analysis

TODO