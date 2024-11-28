set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 SESS_DIR [PROG [TOOLCHAIN [TARGET [BACKEND]]]]"
    exit 1
fi

SESS=$1
PROG=${2:-coremark}
TOOLCHAIN=${3:-gcc}
TARGET=${3:-spike}
BACKEND=${4:-tvmaotplus}

# Create ISAAC session
python3 -m isaac_toolkit.session.create --session $SESS --force

# Run MLonMCU command
python3 -m mlonmcu.cli.main flow run -v --progress $PROG \
    --target $TARGET --backend $BACKEND \
    -f log_instrs -c log_instrs.to_file=1  -c mlif.toolchain=$TOOLCHAIN \
    -c run.export_optional=1 -c mlif.debug_symbols=1 
# Optional: python3 -m mlonmcu.cli.main export --run run/

# Load files
ELF=$MLONMCU_HOME/temp/sessions/latest/runs/latest/generic_mlonmcu
TRACE=$MLONMCU_HOME/temp/sessions/latest/runs/latest/$TARGET_instrs.log

python3 -m isaac_toolkit.frontend.elf.riscv $ELF --session $SESS
python3 -m isaac_toolkit.frontend.instr_trace.$TARGET $TRACE --session $SESS # --operands

# Process files
python3 -m isaac_toolkit.analysis.static.dwarf --session $SESS --force
python3 -m isaac_toolkit.analysis.static.mem_footprint --session $SESS --force
python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $SESS --force
python3 -m isaac_toolkit.analysis.dynamic.trace.instr_operands --session $SESS --force

