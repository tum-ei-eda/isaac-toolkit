# Paths and toolchain
INSTALL_DIR ?= $(abspath ../install)
SESS ?= $(abspath ./sess)
BUILD_DIR ?= $(abspath ./build)
OUT_DIR ?= $(abspath .)
RISCV_PREFIX ?= $(INSTALL_DIR)/rv32im_ilp32
RISCV_NAME ?= riscv32-unknown-elf
RISCV_ARCH ?= rv32im_zicsr_zifencei
RISCV_ABI ?= ilp32
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

OPTIMIZE := 3
EXTRA_COMPILE_FLAGS :=

# ELF / trace file
ELF := $(BUILD_DIR)/$(PROG).elf
MAP := $(BUILD_DIR)/$(PROG).map
DUMP := $(BUILD_DIR)/$(PROG).dump
TRACE := $(OUT_DIR)/$(SIMULATOR)_instrs.log
CALLGRIND_POS = $(OUT_DIR)/callgrind_pos.out
CALLGRIND_PC = $(OUT_DIR)/callgrind_pc.out
CALLGRAPH_DOT = $(OUT_DIR)/callgraph.dot
CALLGRAPH_PDF = $(OUT_DIR)/callgraph.pdf


.PHONY: all clean init compile run \
        load_static load_dynamic load \
        analyze_static analyze_dynamic analyze \
        visualize_static visualize_dynamic visualize \
        profile callgraph kcachegrind

all: init compile run load analyze visualize profile callgraph kcachegrind

clean:
	rm -rf $(BUILD_DIR) $(SESS) *.log *.out callgraph.*

init:
	python3 -m isaac_toolkit.session.create --session $(SESS) $(FORCE_ARG)

compile:
	mkdir -p $(BUILD_DIR)
ifeq ($(SIMULATOR),etiss)
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) \
		$(PROG_SRCS) $(INSTALL_DIR)/etiss/etiss_riscv_examples/riscv_crt0/crt0.S \
		$(INSTALL_DIR)/etiss/etiss_riscv_examples/riscv_crt0/trap_handler.c \
		-T $(INSTALL_DIR)/etiss/install/etiss.ld -nostdlib -lc -lgcc -lsemihost \
		-o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
		-Xlinker -Map=$(MAP)
else
	$(CC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) \
		$(PROG_SRCS) -o $(ELF) $(PROG_INCS) $(PROG_DEFS) -g -O$(OPTIMIZE) \
		-Xlinker -Map=$(MAP)
endif
endif

$(DUMP): $(ELF)
	$(OBJDUMP) -d $(ELF) > $(DUMP)

compile: $(ELF) $(DUMP)

run: $(OUT_DIR)
ifeq ($(SIMULATOR),spike)
	$(SPIKE) --isa=$(RISCV_ARCH)_zicntr -l --log=$(TRACE) $(PK) $(ELF) -s
else ifeq ($(SIMULATOR),etiss)
	$(ETISS) $(ELF) -i$(ETISS_INI) -pPrintInstruction | grep "^0x00000000" > $(TRACE)
endif

load_static: $(ELF)
	python3 -m isaac_toolkit.frontend.elf.riscv --session $(SESS) $(ELF) $(FORCE_ARG)
	@if [ -f "$(MAP)" ]; then python3 -m isaac_toolkit.frontend.linker_map --session $(SESS) $(MAP) $(FORCE_ARG); fi
	@if [ -f "$(DUMP)" ]; then python3 -m isaac_toolkit.frontend.disass.objdump --session $(SESS) $(DUMP) $(FORCE_ARG); fi

load_dynamic: $(TRACE)
	python3 -m isaac_toolkit.frontend.instr_trace.$(SIMULATOR) $(TRACE) --session $(SESS) $(FORCE_ARG)

load: load_static load_dynamic

analyze_static:
	python3 -m isaac_toolkit.analysis.static.dwarf --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.static.mem_footprint --session $(SESS) $(FORCE_ARG)

analyze_dynamic:
	python3 -m isaac_toolkit.analysis.dynamic.histogram.opcode --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.dynamic.histogram.instr --session $(SESS) $(FORCE_ARG)
	python3 -m isaac_toolkit.analysis.dynamic.trace.basic_blocks --session $(SESS) $(FORCE_ARG)

analyze: analyze_static analyze_dynamic

visualize_static:
	python3 -m isaac_toolkit.visualize.pie.mem_footprint --session $(SESS) $(FORCE_ARG)

visualize_dynamic:
	python3 -m isaac_toolkit.visualize.pie.runtime --session $(SESS) --legend $(FORCE_ARG)

visualize: visualize_static visualize_dynamic

profile_pos: $(OUT_DIR)
	python3 -m isaac_toolkit.backend.profile.callgrind --session $(SESS) --dump-pos --output $(CALLGRIND_POS) $(FORCE_ARG)

$(CALLGRIND_POS): profile_pos

profile_pc: $(OUT_DIR)
	python3 -m isaac_toolkit.backend.profile.callgrind --session $(SESS) --dump-pc --output $(CALLGRIND_PC) $(FORCE_ARG)

$(CALLGRIND_PC): profile_pc

profile: profile_pos profile_pc

$(CALLGRAPH_DOT): $(CALLGRIND_PC) $(OUT_DIR)
	gprof2dot --format=callgrind --output=$(CALLGRAPH_DOT) $(CALLGRIND_PC) -n 0.1 -e 0.1 --color-nodes-by-selftime

$(CALLGRAPH_PDF): $(CALLGRAPH_DOT) $(OUT_DIR)
	dot -Tpdf $(CALLGRAPH_DOT) > $(CALLGRAPH_PDF)

callgraph: $(CALLGRAPH_PDF)

# kcachegrind_pos: callgrind_pos.out
# 	@echo "Opening kcachegrind GUI..."
# 	OBJDUMP=$(OBJDUMP) kcachegrind callgrind_pc.out


kcachegrind_pc: $(CALLGRIND_PC)
	@echo "Opening kcachegrind GUI (PC)..."
	OBJDUMP=$(OBJDUMP) kcachegrind $(CALLGRIND_PC)
