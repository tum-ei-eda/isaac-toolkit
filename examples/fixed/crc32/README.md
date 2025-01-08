# CRC32 Minimal Example

For prerequisites (`PYTHONPATH`, venv,...) check other examples!

## Example Usage

```
tar xvf data.tar.xz
export SESS=$(pwd)/sess
python3 -m isaac_toolkit.session.create --session $SESS
python3 -m isaac_toolkit.frontend.elf.riscv --session $SESS data/crc32
python3 -m isaac_toolkit.analysis.static.dwarf --session $SESS
python3 -m isaac_toolkit.frontend.instr_trace.etiss_new --sess $SESS data/asm_trace/asm_trace_*.csv --force
```
