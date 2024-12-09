# isaac-toolkit
Automated Customization Toolkit for Instruction Set Architectures (ISAs)

## Repository Structure

### Python Package

```python
isaac_toolkit
├── algorithm
│   └── ... # work in progress
├── analysis
│   ├── dynamic  # dynamic analysis tools
│   │   ├── histogram
│   │   │   ├── instr.py
│   │   │   ├── opcode.py
│   │   │   └── pc.py
│   │   ├── profile
│   │   │   └── profile.py
│   │   └── trace
│   │       ├── basic_blocks.py
│   │       ├── instr_operands.py
│   │       ├── track_used_functions.py
│   │       └── trunc_trace.py
│   └── static  # static analysis tools
│       ├── dwarf.py
│       ├── histogram
│       │   ├── disass_instr.py
│       │   └── disass_opcode.py
│       ├── linker_map.py
│       ├── llvm_bbs.py
│       └── mem_footprint.py
├── artifact
│   └── artifact.py
├── backend  # ISAAC backends
│   ├── isa  # not implemented
│   │   └── ...
│   ├── ise  # not implemented
│   │   └── ...
│   ├── memgraph  # annotate CFDG database with bb_weights
│   │   └── annotate_bb_weights.py
│   └── profile  # not implemented
│       └── ...
├── frontend  # ISAAC frontends
│   ├── cfg  # configuration parsing
│   │   └── yaml.py
│   ├── disass
│   │   └── objdump.py
│   ├── elf
│   │   └── riscv.py
│   ├── instr_trace
│   │   ├── etiss.py
│   │   └── spike.py
│   ├── isa  # work in progress
│   │   └── ...
│   ├── linker_map.py
│   ├── memgraph  # work in progress
│   │   ├── llvm_ir_cdfg.py
│   │   └── llvm_mir_cdfg.py
│   ├── mem_trace  # not implemented
│   │   └── ...
│   └── source  # not implemented
│       └── ...
├── generate  # ISAAC generators
│   ├── ise  # Propose ISAXes
│   │   ├── choose_bbs.py
│   │   ├── pool
│   │   │   └── random.py
│   │   └── query_candidates_from_db.py
│   └── iss  # ISS retargeting
│       └── generate_etiss_core.py
├── session  # Infrastructure for sessions/artifacts
│   ├── artifact.py
│   ├── config.py
│   ├── create.py
│   ├── session.py
│   └── summary.py
├── utils
│   ├── cli.py
│   ├── nx_converter.py  # Nextworkx graph to images
│   ├── pickle_printer.py  # Convert pickle files (including DFs to text)
│   └── ...
└── visualize
    └── pie  # Pie chart generators
        ├── disass_counts.py
        ├── mem_footprint.py
        └── runtime.py
```

### Examples


## Usage

### Prerequisites

Setup a Python virtual environment:

```sh
virtualenv -p python3 venv/
# Alternative: python3 -m venv venv/
```

Install packages:

```sh
pip install -r requirements.txt

# Optional:
pip install -r requirements_full.txt  # for specific backends
pip install -r requirements_dev.txt  # for linting, testing,...
```

### Demo

~~Package installation: TODO~~

Make sure to add the top level directory of this repository to your Python path:

```sh
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Minimal example:

```sh
python3 -m isaac_toolkit.session.create --session sess/
python3 -m isaac_toolkit.session.summary --session sess/
```

See [`Examples/standalone/coremark/README.md`](Examples/README.md) for an end-to-end example.

## Testing

TODO: add unit & integration tests
