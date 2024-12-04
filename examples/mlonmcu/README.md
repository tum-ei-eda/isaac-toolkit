# MLonMCU Examples

These examples for the ISAAC-Toolkit are the most extensive ones, as several different benchmark programs are supported:

- Embench-IOT
- Taclebench
- Coremark
- Dhyrstone
- MLPerfTiny
- ...

To run these examples, the MLonMCU Tools needs to be installed as explained in [`SETUP.md`](Setup.md).

## Usage

```sh
# ./mlonmcu_example.sh SESS [PROG [TOOLCHAIN [TARGET [BACKEND [EXTRA_ARGS ...]]]]]
# Example Commands:

./mlonmcu_example.sh sess/  # use defaults
./mlonmcu_example.sh sess/ toycar  # override program
./mlonmcu_example.sh sess/ toycar gcc  # override program+toolchain
./mlonmcu_example.sh sess/ toycar gcc etiss  # override program+toolchain+target
./mlonmcu_example.sh sess/ toycar gcc etiss tvmaotplus  # override program+toolchain+target+backend
./mlonmcu_example.sh sess/ toycar gcc etiss tvmaotplus -f muriscvnnbyoc  # override program+toolchain+target+backend+extra_args
```

## Investigate Results

### Static Analysis

Investigate generated tables

```sh
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/file2funcs.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/func2pc.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/pc2locs.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/mem_footprint.pkl | less
```

### Dynamic Analysis

Investigate generated tables

```sh
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/pc2bb.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/instrs_hist.pkl | less
python3 -m isaac_toolkit.utils.pickle_printer $SESS/table/opcodes_hist.pkl | less
```

### Visualizations

Pie charts

```sh
xdg-open $SESS/plots/mem_footprint_per_func.jpg

xdg-open $SESS/plots/runtime_per_func.jpg
xdg-open $SESS/plots/runtime_per_opcode.jpg
xdg-open $SESS/plots/runtime_per_instr.jpg

# Optional:
# xdg-open $SESS/plots/disass_counts_per_instr.jpg
# xdg-open $SESS/plots/disass_counts_per_opcode.jpg
```

### Profiling

*Hint:* The packages `kcachegrind` and `graphviz-dev` need to be installed! Further the `OBJDUMP` environment variable nhas to point to a supported RISC-V objdump program.

```sh
python3 -m isaac_toolkit.backend.profile.callgrind --session sess/ --dump-pos --output callgrind_pos.out
python3 -m isaac_toolkit.backend.profile.callgrind --session sess/ --dump-pc --output callgrind_pc.out

# Callgraph
gprof2dot --format=callgrind --output=callgraph.dot callgrind_pos.out -n 0.1 -e 0.1 --color-nodes-by-selftime
dot -Tpdf callgraph.dot > callgraph.pdf

# Callgrind GUI (instruction level)
OBJDUMP=$OBJDUMP kcachegrind callgrind_pc.out

# Gallgrind GUI (source level)
kcachegrind callgrind_pos.out

# Annotate source code (ASCII)
# callgrind_annotate callgrind_pos.out path/to/src/*.c
```

**Warning:** KCachegrind's machine code view can not handle programs compiled with the RISC-V compressed instructions.

*Workarounds:*
- Do not open the machine code view
- Disable compressed instructions (during compilation)
- Printing the entire disassembly instead of limiting the output to the selected function:

```sh
OBJDUMP_FORMAT="$OBJDUMP -C -d %3 # %1 %2" kcachegrind callgrind_pc.out
```
