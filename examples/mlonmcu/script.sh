set -e

# TODO:
#export RISCV=...
#export LLVM=...
#export MLONMCU_HOME=...
ISAAC_SESSION=$(pwd)/sess

# Create ISAAC session
python3 -m isaac_toolkit.session.create --session $ISAAC_SESSION --force
python3 -m isaac_toolkit.utils.memgraph.purge_db --sess $ISAAC_SESSION

# Run MLonMCU command
# RV32: broken
# python3 -m mlonmcu.cli.main flow run toycar --target etiss --backend tvmaotplus --feature-gen muriscvnnbyoc -c run.export_optional=1  -f log_instrs -c log_instrs.to_file=1 -c session.executor=process_pool -c runs_per_stage=0 -c etiss.compressed=0 -c etiss.atomic=0 -c etiss.fpu=none -c riscv_gcc.install_dir=$RISCV -c llvm.install_dir=$LLVM -c mlif.debug_symbols=1 -v -c mlif.toolchain=llvm -f memgraph_llvm_cdfg -f global_isel -v
# RV64
# python3 -m mlonmcu.cli.main flow run sine_model --target etiss --backend tvmaotplus -c run.export_optional=1 -c log_instrs.to_file=1 -c session.executor=process_pool -c runs_per_stage=0 -c etiss.compressed=0 -c etiss.xlen=64 -c etiss.atomic=0 -c etiss.fpu=double -c riscv_gcc.install_dir=$RISCV -c riscv_gcc.name=riscv64-unknown-elf -c llvm.install_dir=$LLVM -c mlif.debug_symbols=1 -v
python3 -m mlonmcu.cli.main flow run toycar --target etiss --backend tvmaotplus -c run.export_optional=1  -f log_instrs -c log_instrs.to_file=1 -c session.executor=process_pool -c runs_per_stage=0 -c etiss.compressed=0 -c etiss.xlen=64 -c etiss.atomic=0 -c etiss.fpu=double -c riscv_gcc.install_dir=$RISCV -c riscv_gcc.name=riscv64-unknown-elf -c llvm.install_dir=$LLVM -c mlif.debug_symbols=1 -v -c mlif.toolchain=llvm
# python3 -m mlonmcu.cli.main flow run toycar --target etiss --backend tvmaotplus --feature-gen muriscvnnbyoc -c run.export_optional=1  -f log_instrs -c log_instrs.to_file=1 -c session.executor=process_pool -c runs_per_stage=0 -c etiss.compressed=0 -c etiss.xlen=64 -c etiss.atomic=0 -c etiss.fpu=double -c riscv_gcc.install_dir=$RISCV -c riscv_gcc.name=riscv64-unknown-elf -c llvm.install_dir=$LLVM -c mlif.debug_symbols=1 -v -c mlif.toolchain=llvm
# -f memgraph_llvm_cdfg -f global_isel -v


# Load files
ELF=$MLONMCU_HOME/temp/sessions/latest/runs/latest/generic_mlonmcu
TRACE=$MLONMCU_HOME/temp/sessions/latest/runs/latest/etiss_instrs.log

python3 -m isaac_toolkit.frontend.elf.riscv $ELF --session $ISAAC_SESSION
python3 -m isaac_toolkit.frontend.instr_trace.etiss $TRACE --session $ISAAC_SESSION
python3 -m isaac_toolkit.frontend.memgraph.llvm_mir_cdfg --session sess -f
# TODO: python3 -m isaac_toolkit.frontend.isa.m2isar /work/git/dlr/etiss_arch_riscv/gen_model/top.m2isarmodel --session $ISAAC_SESSION -f

# Process files
python3 -m isaac_toolkit.analysis.static.dwarf --session $ISAAC_SESSION --force
python3 -m isaac_toolkit.analysis.static.mem_footprint --session $ISAAC_SESSION --force
python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $ISAAC_SESSION --force
python3 -m isaac_toolkit.analysis.dynamic.trace.instr_operands --session $ISAAC_SESSION --force


# Investigate artifacts
python3 -m isaac_toolkit.session.summary --session $ISAAC_SESSION
python3 -m isaac_toolkit.utils.pickle_printer sess/table/func2pc.pkl
# TODO: python3 -m isaac_toolkit.utils.nx_converter $ISAAC_SESSION/graph/memgraph_mir_cdfg.pkl $ISAAC_SESSION/graph/memgraph_mir_cdfg.pdf
# ...
