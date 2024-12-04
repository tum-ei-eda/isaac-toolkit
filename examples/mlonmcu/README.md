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
# TODO
SESS_DIR=$(pwd)/sess
PROG_NAME=toycar
# ./mlonmcu_example.sh SESS_DIR [PROG [TOOLCHAIN [TARGET [BACKEND]]]]
# i.e.:
./mlonmcu_example.sh sess/ toycar ...
```
