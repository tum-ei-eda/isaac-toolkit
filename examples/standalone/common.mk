# Paths and toolchain
INSTALL_DIR ?= $(abspath ../install)
SESS ?= $(abspath ./sess)
BUILD_DIR ?= $(abspath ./build)
OUT_DIR ?= $(abspath ./out)
RISCV_PREFIX ?= $(INSTALL_DIR)/rv32im_ilp32
RISCV_NAME ?= riscv32-unknown-elf
RISCV_ARCH ?= rv32im_zicsr_zifencei
RISCV_ABI ?= ilp32
# TODO: RISCV_CMODEL
SYSROOT ?= $(RISCV_PREFIX)/$(RISCV_NAME)
CC := $(RISCV_PREFIX)/bin/$(RISCV_NAME)-gcc
OBJDUMP := $(RISCV_PREFIX)/bin/$(RISCV_NAME)-objdump


FORCE ?= 0
FORCE_ARG := $(if $(filter 1,$(FORCE)),--force,)

# Simulation
SIMULATOR ?= spike
SPIKE ?= $(INSTALL_DIR)/spike/spike
PK ?= $(INSTALL_DIR)/spike/pk_rv32gc
ETISS ?= $(INSTALL_DIR)/etiss/install/bin/run_helper.sh
ETISS_INI ?= $(INSTALL_DIR)/etiss/install/custom.ini

TGC_BSP_DIR ?= /path/to/tgc/bsp
TGC_SRC_DIR ?= /path/to/tgc/src/dir
TGC_BUILD_DIR ?= $(TGC_SRC_DIR)/build
TGC_INSTALL_DIR ?= $(TGC_SRC_DIR)/install

ifeq ($(SIMULATOR),tgc)
  ifneq ($(wildcard $(TGC_INSTALL_DIR)),)
    TGC_SIM     := $(TGC_INSTALL_DIR)/bin/tgc-sim
    TGC_PCTRACE := $(TGC_INSTALL_DIR)/libexec/pctrace.so
    TGC_YAML    := $(TGC_INSTALL_DIR)/share/tgc-vp/TGC5C_instr.yaml
    TRANSFORM_TRACE_SCRIPT := $(TGC_INSTALL_DIR)/bin/transform_trc
    # GEN_MERMAID_SCRIPT := $(TGC_INSTALL_DIR)/bin/gen_mermaid
    GEN_FLAMEGRAPH_SCRIPT := $(TGC_INSTALL_DIR)/bin/gen_flamegraph
    FLAMEGRAPH_PL := $(TGC_INSTALL_DIR)/bin/flamegraph.pl
    GEN_CACHEGRIND_SCRIPT := $(TGC_INSTALL_DIR)/bin/gen_cachegrind
    TRC2LCOV_SCRIPT := $(TGC_INSTALL_DIR)/bin/trc2lcov

  else ifneq ($(wildcard $(TGC_BUILD_DIR)),)
    TGC_SIM     := $(TGC_BUILD_DIR)/dbt-rise-tgc/tgc-sim
    TGC_PCTRACE := $(TGC_BUILD_DIR)/dbt-rise-plugins/pctrace/pctrace.so
    TGC_YAML    := $(TGC_SRC_DIR)/dbt-rise-tgc/contrib/instr/TGC5C_instr.yaml
    TRANSFORM_TRACE_SCRIPT := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/transform_trc.py
    GEN_MERMAID_SCRIPT := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/gen_mermaid.py
    GEN_FLAMEGRAPH_SCRIPT := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/gen_flamegraph.py
    FLAMEGRAPH_PL := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/flamegraph.pl
    GEN_CACHEGRIND_SCRIPT := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/gen_cachegrind.py
    TRC2LCOV_SCRIPT := $(TGC_SRC_DIR)/dbt-rise-plugins/scripts/trc2lcov.py
  else
    $(error Neither TGC_INSTALL_DIR ($(TGC_INSTALL_DIR)) nor TGC_BUILD_DIR ($(TGC_BUILD_DIR)) exists!)
  endif
endif

OPTIMIZE ?= 2
EXTRA_COMPILE_FLAGS ?=

# ELF / trace file
ELF := $(BUILD_DIR)/$(PROG).elf
MAP := $(BUILD_DIR)/$(PROG).map
DUMP := $(BUILD_DIR)/$(PROG).dump
TRACE := $(OUT_DIR)/$(SIMULATOR)_instrs.log
OUTP := $(OUT_DIR)/$(SIMULATOR)_out.log
TIMING_CSV := $(OUT_DIR)/stage_timings.csv
CALLGRIND_POS = $(OUT_DIR)/callgrind_pos.out
CALLGRIND_PC = $(OUT_DIR)/callgrind_pc.out
CALLGRAPH_DOT = $(OUT_DIR)/callgraph.dot
CALLGRAPH_PDF = $(OUT_DIR)/callgraph.pdf

REPORT_FMT ?= "html"
REPORT_TOPK ?= 10

define time_stage
@echo "Starting $(1)..."
@start=$$(date +%s%3N); \
$(2); \
end=$$(date +%s%3N); \
elapsed=$$((end - start)); \
echo "$(1),$$elapsed" >> $(TIMING_CSV); \
echo "Finished $(1) in $$elapsed ms"
endef


.PHONY: all clean init compile run \
        load_static load_dynamic load \
        analyze_static analyze_dynamic analyze \
        visualize_static visualize_dynamic visualize \
        profile profile_pc profile_pos \
        callgraph kcachegrind kcachegrind_pc kcachegrind_pos \
	function_trace flamegraph mermaid cachegrind lcov \
	measure measure_flow measure_full_flow flow_load \
	flow_analyze flow_visualize flow_profile

all: init compile run load analyze visualize profile callgraph

measure: $(OUT_DIR)
	@echo "stage,time_ms" > $(TIMING_CSV)
	$(call time_stage,init, $(MAKE) init)
	$(call time_stage,compile, $(MAKE) compile)
	$(call time_stage,run, $(MAKE) run)
	$(call time_stage,trace, $(MAKE) trace)
	$(call time_stage,load_static, $(MAKE) load_static)
	$(call time_stage,load_dynamic, $(MAKE) load_dynamic)
	$(call time_stage,analyze_static, $(MAKE) analyze_static)
	$(call time_stage,analyze_dynamic, $(MAKE) analyze_dynamic)
	$(call time_stage,visualize_static, $(MAKE) visualize_static)
	$(call time_stage,visualize_dynamic, $(MAKE) visualize_dynamic)
	$(call time_stage,report, $(MAKE) report)
	$(call time_stage,profile_pos, $(MAKE) profile_pos)
	$(call time_stage,profile_pc, $(MAKE) profile_pc)
	$(call time_stage,callgraph, $(MAKE) callgraph)
	# $(call time_stage,kcachegrind_pc, $(MAKE) kcachegrind_pc)
	# $(call time_stage,kcachegrind_pos, $(MAKE) kcachegrind_pos)
	$(call time_stage,function_trace, $(MAKE) function_trace)
	$(call time_stage,flamegraph, $(MAKE) flamegraph)
	$(call time_stage,cachegrind, $(MAKE) cachegrind)
	$(call time_stage,lcov, $(MAKE) lcov)
# TODO: report

measure_flow: $(OUT_DIR)
	@echo "stage,time_ms" > $(TIMING_CSV)
	$(call time_stage,init, $(MAKE) init)
	$(call time_stage,compile, $(MAKE) compile)
	$(call time_stage,run, $(MAKE) run)
	$(call time_stage,trace, $(MAKE) trace)
	$(call time_stage,flow_load, $(MAKE) flow_load)
	$(call time_stage,flow_analyze, $(MAKE) flow_analyze)
	$(call time_stage,flow_visualize, $(MAKE) flow_visualize)
	$(call time_stage,flow_report, $(MAKE) flow_report)
	$(call time_stage,flow_profile, $(MAKE) flow_profile)
	$(call time_stage,callgraph, $(MAKE) callgraph)
	# $(call time_stage,kcachegrind_pc, $(MAKE) kcachegrind_pc)
	# $(call time_stage,kcachegrind_pos, $(MAKE) kcachegrind_pos)
	$(call time_stage,function_trace, $(MAKE) function_trace)
	$(call time_stage,flamegraph, $(MAKE) flamegraph)
	$(call time_stage,cachegrind, $(MAKE) cachegrind)
	$(call time_stage,lcov, $(MAKE) lcov)

measure_full_flow: $(OUT_DIR)
	@echo "stage,time_ms" > $(TIMING_CSV)
	$(call time_stage,init, $(MAKE) init)
	$(call time_stage,compile, $(MAKE) compile)
	$(call time_stage,run, $(MAKE) run)
	$(call time_stage,trace, $(MAKE) trace)
	$(call time_stage,full_flow, $(MAKE) full_flow)
	$(call time_stage,callgraph, $(MAKE) callgraph)
	# $(call time_stage,kcachegrind_pc, $(MAKE) kcachegrind_pc)
	# $(call time_stage,kcachegrind_pos, $(MAKE) kcachegrind_pos)
	$(call time_stage,function_trace, $(MAKE) function_trace)
	$(call time_stage,flamegraph, $(MAKE) flamegraph)
	$(call time_stage,cachegrind, $(MAKE) cachegrind)
	$(call time_stage,lcov, $(MAKE) lcov)

clean:
	rm -rf $(BUILD_DIR) $(SESS) *.log *.out $(CALLGRAPH_DOT) $(CALLGRAPH_PDF) $(CALLGRIND_POS) $(CALLGRIND_PC) $(TRACE)

$(SESS):
	python3 -m isaac_toolkit.session.create --session $(SESS) $(FORCE_ARG)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

init: $(SESS)

ifeq ($(SIMULATOR),tgc)
# LIBWRAP = $(BUILD_DIR)/libwrap.a
# $(LIBWRAP): $(LIBWRAP_OBJS)
include $(TGC_BSP_DIR)/libwrap/libwrap.mk
$(ELF): $(PROG_SRCS) $(LIBWRAP)
	mkdir -p $(BUILD_DIR)
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) \
		$(PROG_SRCS) $(TGC_BSP_DIR)/env/start.S $(TGC_BSP_DIR)/env/entry.S \
		$(TGC_BSP_DIR)/env/iss/init.c \
		$(TGC_BSP_DIR)/env/iss/bsp_write.c \
    -L$(TGC_BSP_DIR)/env/ \
    -T$(TGC_BSP_DIR)/env//iss/link.lds \
    -nostartfiles \
    -I$(TGC_BSP_DIR)/include -I$(TGC_BSP_DIR)/drivers/ -I$(TGC_BSP_DIR)/env/ -I$(TGC_BSP_DIR)/env/iss -I$(TGC_BSP_DIR)/libwrap/sys/ \
    -DBOARD_iss \
    -Wl,--wrap=printf -Wl,--wrap=malloc -Wl,--wrap=open -Wl,--wrap=lseek -Wl,--wrap=_lseek -Wl,--wrap=read -Wl,--wrap=_read -Wl,--wrap=write -Wl,--wrap=_write -Wl,--wrap=fstat -Wl,--wrap=_fstat -Wl,--wrap=stat -Wl,--wrap=close -Wl,--wrap=_close -Wl,--wrap=link -Wl,--wrap=unlink -Wl,--wrap=execve -Wl,--wrap=fork -Wl,--wrap=getpid -Wl,--wrap=kill -Wl,--wrap=wait -Wl,--wrap=isatty -Wl,--wrap=times -Wl,--wrap=sbrk -Wl,--wrap=_sbrk -Wl,--wrap=exit -Wl,--wrap=_exit -Wl,--wrap=puts -Wl,--wrap=_puts -Wl,--wrap=printf -Wl,--wrap=sprintf -L. -Wl,--start-group -lwrap -lc -Wl,--end-group \
    -Wl,--no-warn-rwx-segments \
    $(EXTRA_COMPILE_FLAGS) \
		-o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
		-Xlinker -Map=$(MAP)
else
$(ELF): $(PROG_SRCS)
	mkdir -p $(BUILD_DIR)
ifeq ($(SIMULATOR),etiss)
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) \
		$(PROG_SRCS) $(INSTALL_DIR)/etiss/etiss_riscv_examples/riscv_crt0/crt0.S \
		$(INSTALL_DIR)/etiss/etiss_riscv_examples/riscv_crt0/trap_handler.c \
		-T $(INSTALL_DIR)/etiss/install/etiss.ld -nostdlib -lc -lgcc -lsemihost \
		-o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
    $(EXTRA_COMPILE_FLAGS) \
		-Xlinker -Map=$(MAP)
else ifeq ($(SIMULATOR),spike_bm)
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) -specs=htif_nano.specs -specs=htif_wrap.specs \
		$(PROG_SRCS) -o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
    $(EXTRA_COMPILE_FLAGS) \
		-Xlinker -Map=$(MAP)
else
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) \
		$(PROG_SRCS) -o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
    $(EXTRA_COMPILE_FLAGS) \
		-Xlinker -Map=$(MAP)
endif
endif

$(DUMP): $(ELF)
	$(OBJDUMP) -d $(ELF) > $(DUMP)

compile: $(ELF) $(DUMP)

$(TRACE): $(ELF) | $(OUT_DIR)
ifeq ($(SIMULATOR),spike)
	$(SPIKE) --isa=$(RISCV_ARCH)_zicntr -l --log=$(TRACE) $(PK) $(ELF) -s
else ifeq ($(SIMULATOR),spike_bm)
	$(SPIKE) --isa=$(RISCV_ARCH)_zicntr -l --log=$(TRACE) $(ELF) -s
else ifeq ($(SIMULATOR),etiss)
	# $(ETISS) $(ELF) -i$(ETISS_INI) -pPrintInstruction | grep "^0x00000000" > $(TRACE)
	$(ETISS) $(ELF) -i$(ETISS_INI) -pPrintInstruction --plugin.printinstruction.print_to_file=true --etiss.output_path_prefix=$(OUT_DIR)
	mv $(OUT_DIR)/instr_trace.csv $(TRACE)
else ifeq ($(SIMULATOR),tgc)
	$(TGC_SIM) -f $(ELF) -p $(TGC_PCTRACE)=$(TGC_YAML)
	mv output.trc $(TRACE)
endif

$(OUTP): $(ELF) | $(OUT_DIR)
ifeq ($(SIMULATOR),spike)
	$(SPIKE) --isa=$(RISCV_ARCH)_zicntr $(PK) $(ELF) -s | tee $(OUTP)
else ifeq ($(SIMULATOR),spike_bm)
	$(SPIKE) --isa=$(RISCV_ARCH)_zicntr $(ELF) -s | tee $(OUTP)
else ifeq ($(SIMULATOR),etiss)
	$(ETISS) $(ELF) -i$(ETISS_INI) | tee $(OUTP)
else ifeq ($(SIMULATOR),tgc)
	$(TGC_SIM) -f $(ELF) | tee $(OUTP)
endif

trace: $(TRACE)
run: $(OUTP)

full_flow: $(ELF) $(MAP) $(DUMP) $(TRACE)
	python3 -m isaac_toolkit.flow.rvf.full_flow --session $(SESS) --elf $(ELF) --linker-map $(MAP) --disass $(DUMP) --instr-trace $(TRACE) --report-fmt $(REPORT_FMT) --report-detailed --report-portable --report-style --report-topk $(REPORT_TOPK) $(FORCE_ARG)
	cp $(SESS)/profile/callgrind_pc.out $(CALLGRIND_PC)
	cp $(SESS)/profile/callgrind_pos.out $(CALLGRIND_POS)

flow_load: $(ELF) $(MAP) $(DUMP) $(TRACE)
	python3 -m isaac_toolkit.flow.rvf.stage.load --session $(SESS) --elf $(ELF) --linker-map $(MAP) --disass $(DUMP) --instr-trace $(TRACE)	$(FORCE_ARG)

load_static: $(ELF)
	python3 -m isaac_toolkit.frontend.elf.riscv --session $(SESS) $(ELF) $(FORCE_ARG)
	@if [ -f "$(MAP)" ]; then python3 -m isaac_toolkit.frontend.linker_map --session $(SESS) $(MAP) $(FORCE_ARG); fi
	@if [ -f "$(DUMP)" ]; then python3 -m isaac_toolkit.frontend.disass.objdump --session $(SESS) $(DUMP) $(FORCE_ARG); fi

ifeq ($(SIMULATOR),spike_bm)
INSTR_TRACE_FRONTEND ?= spike
else
INSTR_TRACE_FRONTEND ?= $(SIMULATOR)
endif

load_dynamic: $(TRACE)
	python3 -m isaac_toolkit.frontend.instr_trace.$(INSTR_TRACE_FRONTEND) $(TRACE) --session $(SESS) $(FORCE_ARG)

load: load_static load_dynamic

flow_analyze:
	python3 -m isaac_toolkit.flow.rvf.stage.analyze --session $(SESS)

analyze_static:
	python3 -m isaac_toolkit.analysis.static.dwarf --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.static.mem_footprint --session $(SESS) $(FORCE_ARG)

analyze_dynamic:
	python3 -m isaac_toolkit.analysis.dynamic.histogram.opcode --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.dynamic.histogram.instr --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $(SESS) $(FORCE_ARG)

analyze: analyze_static analyze_dynamic

flow_visualize:
	python3 -m isaac_toolkit.flow.rvf.stage.visualize --session $(SESS) $(FORCE_ARG)

# TODO: visualize diass counts?
visualize_static:
	python3 -m isaac_toolkit.visualize.pie.mem_footprint --session $(SESS) --legend $(FORCE_ARG)

visualize_dynamic:
	python3 -m isaac_toolkit.visualize.pie.runtime --session $(SESS) --legend $(FORCE_ARG)

visualize: visualize_static visualize_dynamic

flow_report:
	python3 -m isaac_toolkit.flow.rvf.stage.report --session $(SESS) $(FORCE_ARG) --fmt $(REPORT_FMT) --detailed --portable --style --topk $(REPORT_TOPK)

report:
	python3 -m isaac_toolkit.report.report_runtime --session $(SESS) $(FORCE_ARG) --fmt $(REPORT_FMT) --detailed --portable --style --topk $(REPORT_TOPK)

flow_profile:
	python3 -m isaac_toolkit.flow.rvf.stage.profile --session $(SESS) $(FORCE_ARG)
	cp $(SESS)/profile/callgrind_pc.out $(CALLGRIND_PC)
	cp $(SESS)/profile/callgrind_pos.out $(CALLGRIND_POS)

$(CALLGRIND_POS): | $(OUT_DIR)
	python3 -m isaac_toolkit.backend.profile.callgrind --session $(SESS) --dump-pos --output $(CALLGRIND_POS) $(FORCE_ARG)

$(CALLGRIND_PC): | $(OUT_DIR)
	python3 -m isaac_toolkit.backend.profile.callgrind --session $(SESS) --dump-pc --output $(CALLGRIND_PC) $(FORCE_ARG)

profile_pc: $(CALLGRIND_PC)
profile_pos: $(CALLGRIND_POS)

$(CALLGRAPH_DOT): $(CALLGRIND_PC) | $(OUT_DIR)
	gprof2dot --format=callgrind --output=$(CALLGRAPH_DOT) $(CALLGRIND_PC) -n 0.1 -e 0.1 --color-nodes-by-selftime

$(CALLGRAPH_PDF): $(CALLGRAPH_DOT) | $(OUT_DIR)
	dot -Tpdf $(CALLGRAPH_DOT) > $(CALLGRAPH_PDF)

callgraph: $(CALLGRAPH_PDF)

# kcachegrind_pos: callgrind_pos.out
# 	@echo "Opening kcachegrind GUI..."
# 	OBJDUMP=$(OBJDUMP) kcachegrind callgrind_pc.out

# kcachegrind_pc: $(CALLGRIND_PC)
# 	@echo "Opening kcachegrind GUI (PC)..."
# 	OBJDUMP=$(OBJDUMP) kcachegrind $(CALLGRIND_PC)

ifeq ($(SIMULATOR),tgc)
FUNCTION_TRACE_JSON := $(OUT_DIR)/function_trace.json

$(FUNCTION_TRACE_JSON): $(ELF) $(TRACE) | $(OUT_DIR)
	python3 $(TRANSFORM_TRACE_SCRIPT) $(ELF) $(TRACE) -output $(FUNCTION_TRACE_JSON)

function_trace: $(FUNCTION_TRACE_JSON)

MERMAID_OUT := $(OUT_DIR)/mermaid.out

$(MERMAID_OUT): $(FUNCTION_TRACE_JSON) | $(OUT_DIR)
	cd $(OUT_DIR) && python3 $(GEN_MERMAID_SCRIPT) > $(MERMAID_OUT)
mermaid: $(MERMAID_OUT)

FLAMEGRAPH_IN := $(OUT_DIR)/flamegraph.in

$(FLAMEGRAPH_IN): $(FUNCTION_TRACE_JSON) | $(OUT_DIR)
	cd $(OUT_DIR) && python3 $(GEN_FLAMEGRAPH_SCRIPT)

FLAMEGRAPH_SVG := $(OUT_DIR)/flamegraph.svg

$(FLAMEGRAPH_SVG): $(FLAMEGRAPH_IN) | $(OUT_DIR)
	perl $(FLAMEGRAPH_PL) $(FLAMEGRAPH_IN) > $(FLAMEGRAPH_SVG)

flamegraph: $(FLAMEGRAPH_SVG)

CACHEGRIND_IN := $(OUT_DIR)/cachegrind.in

$(CACHEGRIND_IN): $(FUNCTION_TRACE_JSON) | $(OUT_DIR)
	cd $(OUT_DIR) && python3 $(GEN_CACHEGRIND_SCRIPT)

cachegrind: $(CACHEGRIND_IN)

LCOV_OUT := $(OUT_DIR)/coverage.info
LCOV_HTML := $(OUT_DIR)/html

# NEEDS sudo apt install lcov
$(LCOV_HTML): $(ELF) $(TRACE) | $(OUT_DIR)
	python3 $(TRC2LCOV_SCRIPT) $(ELF) --trc $(TRACE) --output $(LCOV_OUT) --genhtml $(LCOV_HTML)

$(LCOV_OUT): $(LCOV_HTML)

lcov: $(LCOV_HTML)


endif
