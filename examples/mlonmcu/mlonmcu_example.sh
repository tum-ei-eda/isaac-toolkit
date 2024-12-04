set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 SESS_DIR [PROG [TOOLCHAIN [TARGET [BACKEND]]]]"
    exit 1
fi

SESS=$1
PROG=${2:-toycar}
TOOLCHAIN=${3:-gcc}
TARGET=${3:-etiss}
BACKEND=${4:-tvmaotplus}

# Create ISAAC session
python3 -m isaac_toolkit.session.create --session $SESS --force

# Run MLonMCU command
python3 -m mlonmcu.cli.main flow run -v --progress $PROG \
    --target $TARGET --backend $BACKEND \
    -f log_instrs -c log_instrs.to_file=1  -c mlif.toolchain=$TOOLCHAIN \
    -c run.export_optional=1 -c mlif.debug_symbols=1 -c etiss.arch=rv32im_zicsr_zifencei riscv_gcc.install_dir=/work/git/isaac-toolkit/examples/standalone/install/rv32im_ilp32/
# Optional: python3 -m mlonmcu.cli.main export --run run/

# Load files
ELF=$MLONMCU_HOME/temp/sessions/latest/runs/latest/generic_mlonmcu
TRACE=$MLONMCU_HOME/temp/sessions/latest/runs/latest/${TARGET}_instrs.log
MAP=$MLONMCU_HOME/mlif/generic/linker.map
DUMP=$MLONMCU_HOME/generic_mlonmcu.dump

# Load artifacts
python3 -m isaac_toolkit.frontend.elf.riscv $ELF --session $SESS --force
python3 -m isaac_toolkit.frontend.instr_trace.$TARGET $TRACE --session $SESS --force # --operands

# Optional:
# python3 -m isaac_toolkit.frontend.linker_map --session $SESS $MAP --force
# python3 -m isaac_toolkit.frontend.disass.objdump --session $SESS $DUMP --force

# Static analysis
python3 -m isaac_toolkit.analysis.static.dwarf --session $SESS --force
python3 -m isaac_toolkit.analysis.static.mem_footprint --session $SESS --force

# Optional:
# python3 -m isaac_toolkit.analysis.static.linker_map --session $SESS --force
# python3 -m isaac_toolkit.analysis.static.histogram.disass_instr --session $SESS --force
# python3 -m isaac_toolkit.analysis.static.histogram.disass_opcode --session $SESS --force

# Dynamic analysis
python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $SESS --force

# Optional:
# python3 -m isaac_toolkit.analysis.dynamic.trace.instr_operands --session $SESS --force
# python3 -m isaac_toolkit.analysis.dynamic.trace.track_used_functions --session $SESS --force

# Profiling
python3 -m isaac_toolkit.backend.profile.callgrind --session $SESS --dump-pos --output callgrind_pos.out
python3 -m isaac_toolkit.backend.profile.callgrind --session $SESS --dump-pc --output callgrind_pc.out

# Callgraph
gprof2dot --format=callgrind --output=callgraph.dot callgrind_pos.out -n 0.1 -e 0.1 --color-nodes-by-selftime
dot -Tpdf callgraph.dot > callgraph.pdf
