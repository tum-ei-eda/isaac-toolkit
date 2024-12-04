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

TODO

### Dynamic Analysis

TODO

### Profiling

**Warning:** KCachegrind's machine code view can not handle programs compiled with the RISC-V compressed instructions.

*Workarounds:*
- Do not open the machine code view
- Disable compressed instructions (during compilation)
- Printing the entire disassembly instead of limiting the output to the selected function:

```sh
OBJDUMP_FORMAT="/bin/riscv32-unknown-elf-objdump -C -d %3 # %1 %2" kcachegrind callgrind_pc.out
```
