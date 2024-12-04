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
export RISCV_NAME=riscv32-unknown-elf
export RISCV_ARCH=rv32im_zicsr_zifencei
export RISCV_ABI=ilp32
export SYSROOT=$RISCV_PREFIX/$RISCV_NAME
export CC=$RISCV_PREFIX/bin/$RISCV_NAME-gcc
export OBJDUMP=$RISCV_PREFIX/bin/$RISCV_NAME-objdump

# For simulation
# spike
export SIMULATOR=spike
export SPIKE=$INSTALL_DIR/spike/spike
export PK=$INSTALL_DIR/spike/pk_rv32gc
# etiss
# export SIMULATOR=etiss
# export ETISS=$INSTALL_DIR/etiss/install/bin/run_helper.sh
# export ETISS_CRT=$INSTALL_DIR/etiss/etiss_riscv_examples/riscv_crt0
# export ETISS_LDSCRIPT=$INSTALL_DIR/etiss/install/etiss.ld
# export ETISS_INI=$INSTALL_DIR/etiss/install/custom.ini
```

For the profiling step, additional Python packages should be installed:

```sh
pip install -r requirements.txt
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
# cmake -S . -B $BUILD_DIR -DSIMULATOR=$SIMMULATOR
# cmake --build $BUILD_DIR
```

#### Via Makefile

TODO
```sh
# make
```

#### Custom (manual)

##### GCC

```sh
# spike
mkdir -p $BUILD_DIR
$CC -march=$RISCV_ARCH -mabi=$RISCV_ABI src/*.c -o $BUILD_DIR/coremark.elf -Iinc/ -DITERATIONS=100 -DFLAGS_STR='"testing"' -DPERFORMANCE_RUN -DHAS_STDIO -g -O3 -Xlinker -Map=build/coremark.map
$OBJDUMP -d $BUILD_DIR/coremark.elf > $BUILD_DIR/coremark.dump

# etiss
# $CC -march=$RISCV_ARCH -mabi=$RISCV_ABI src/*.c $ETISS_CRT/crt0.S $ETISS_CRT/trap_handler.c -T $ETISS_LDSCRIPT -nostdlib -lc -lgcc -lsemihost -o $BUILD_DIR/coremark.elf -Iinc/ -DITERATIONS=100 -DFLAGS_STR='"testing"' -DPERFORMANCE_RUN -DHAS_STDIO -g -O3 -Xlinker -Map=build/coremark.map
```

##### LLVM

TODO
```sh
...
```

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

#### Via CMake

TODO
```sh
cmake --build $BUILD_DIR --target run
```

#### Via Makefile

TODO
```sh
# make run
```

#### Manual

##### Spike

```sh
$SPIKE --isa=${ARCH}_zicntr -l --log=spike_instrs.log $PK $BUILD_DIR/coremark.elf -s
```

##### ETISS

```sh
$ETISS build/coremark.elf -i$ETISS_INI -pPrintInstruction | grep "^0x00000000" > etiss_instrs.log

# Optional (without trace)
# $ETISS build/coremark.elf -i$ETISS_INI
```

### 4. Dynamic Analysis

Load simulation artifacts into ISAAC Session:

```sh
python3 -m isaac_toolkit.frontend.instr_trace.${SIMULATOR} ${SIMULATOR}_instrs.log --session $SESS
# Optional argument (for supported simulators): --operands
```

Run dynamic analysis steps:

```sh

python3 -m isaac_toolkit.analysis.dynamic.histogram.opcode --session $SESS
python3 -m isaac_toolkit.analysis.dynamic.histogram.instr --session $SESS
python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $SESS

# Optional
# python3 -m isaac_toolkit.analysis.dynamic.trace.instr_operands --session $SESS --imm-only
# python3 -m isaac_toolkit.analysis.dynamic.trace.track_used_functions --session $SESS
```

Generate visualizations:

```sh
python3 -m isaac_toolkit.visualize.pie.runtime --session $SESS --legend

# Optional:
# For effective memory footprint
python3 -m isaac_toolkit.visualize.pie.mem_footprint --session $SESS --legend --force
```

Investigate generated tables

```sh
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/pc2bb.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/instrs_hist.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/opcodes_hist.pkl | less
```

Investigate pie charts:

```sh
xdg-open $SESS/plots/runtime_per_func.jpg
xdg-open $SESS/plots/runtime_per_opcode.jpg
xdg-open $SESS/plots/runtime_per_instr.jpg
```

Optional: Profiling (WIP)

*Hint:* The packages `kcachegrind` and `graphviz-dev` need to be installed!

```sh
python3 -m isaac_toolkit.backend.profile.callgrind --session $SESS --dump-pos --output callgrind_pos.out
python3 -m isaac_toolkit.backend.profile.callgrind --session $SESS --dump-pc --output callgrind_pc.out

# Callgraph
gprof2dot --format=callgrind --output=callgraph.dot callgrind_pos.out -n 0.1 -e 0.1 --color-nodes-by-selftime
dot -Tpdf callgraph.dot > callgraph.pdf

# Callgrind GUI (instruction level)
OBJDUMP=$OBJDUMP kcachegrind callgrind_pc.out

# Gallgrind GUI (source level)
kcachegrind callgrind_pos.out

# Annotate source code (ASCII)
callgrind_annotate callgrind_pos.out src/*.c
```

**Warning:** KCachegrind's machine code view can not handle programs compiled with the RISC-V compressed instructions.

*Workarounds:*
- Do not open the machine code view
- Disable compressed instructions (during compilation)
- Printing the entire disassembly instead of limiting the output to the selected function:

```sh
OBJDUMP_FORMAT="$OBJDUMP -C -d %3 # %1 %2" kcachegrind callgrind_pc.out
```
