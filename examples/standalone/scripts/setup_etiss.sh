#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 DEST [ETISS_REF [BUILD_TYPE]]"
    exit 1
fi

ETISS_DIR=$(readlink -f $1)
ETISS_REF=${2:-e55e2c474013be6af44789e65699d13e14c7aab0}
CMAKE_BUILD_TYPE=${3:-Release}

ETISS_EXAMPLES_DIR=$ETISS_DIR/etiss_riscv_examples
ETISS_BUILD_DIR=$ETISS_DIR/build
ETISS_INSTALL_DIR=$ETISS_DIR/install
ETISS_INI=$ETISS_INSTALL_DIR/custom.ini
ETISS_LDSCRIPT=$ETISS_INSTALL_DIR/etiss.ld


NPROC=$(nproc)

if [[ -d $ETISS_DIR ]]
then
    echo "ETISS already cloned!"
else
    git clone https://github.com/tum-ei-eda/etiss.git $ETISS_DIR
fi

if [[ -d $ETISS_EXAMPLES_DIR ]]
then
    echo "ETISS examples already cloned!"
else
    git clone https://github.com/tum-ei-eda/etiss_riscv_examples.git $ETISS_EXAMPLES_DIR
fi

git -C $ETISS_DIR checkout $ETISS_REF

mkdir -p $ETISS_BUILD_DIR

cmake -B $ETISS_BUILD_DIR -S $ETISS_DIR -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_INSTALL_PREFIX:PATH=$ETISS_INSTALL_DIR

cmake --build $ETISS_BUILD_DIR -j$NPROC
cmake --install $ETISS_BUILD_DIR

# TODO: call setup_etiss_min.sh

export MEM_ROM_ORIGIN=0x10000000
export MEM_ROM_LENGTH=0x00400000
export MEM_RAM_ORIGIN=0x20000000
export MEM_RAM_LENGTH=0x00100000

export MIN_STACK_SIZE=0x1000
export MIN_HEAP_SIZE=0x1000

cat <<EOT > $ETISS_INI
[StringConfigurations]
etiss.output_path_prefix=
jit.type=TCCJIT

[BoolConfigurations]
arch.or1k.ignore_sr_iee=false
jit.gcc.cleanup=true
jit.verify=false
etiss.load_integrated_libraries=true
jit.debug=false
etiss.enable_dmi=false
etiss.log_pc=false

[IntConfigurations]
arch.or1k.if_stall_cycles=0
etiss.max_block_size=100
arch.cpu_cycle_time_ps=31250
ETISS::CPU_quantum_ps=100000
ETISS::write_pc_trace_from_time_us=0
ETISS::write_pc_trace_until_time_us=3000000
ETISS::sim_mode=0
vp::simulation_time_us=20000000
etiss.loglevel=4

[BoolConfigurations]
arch.enable_semihosting=true

[IntConfigurations]
simple_mem_system.memseg_origin_00=$MEM_ROM_ORIGIN
simple_mem_system.memseg_length_00=$MEM_ROM_LENGTH
simple_mem_system.memseg_origin_01=$MEM_RAM_ORIGIN
simple_mem_system.memseg_length_01=$MEM_RAM_LENGTH
EOT

cat <<EOT > $ETISS_LDSCRIPT
/*
// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the “License”); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// This file was modified by the Chair of Electronic Design Automation, TUM
*/

ENTRY(_start)
SEARCH_DIR(.)


MEMORY
{
  ROM  (rx)  : ORIGIN = $MEM_ROM_ORIGIN, LENGTH = $MEM_ROM_LENGTH
  RAM  (rw) : ORIGIN = $MEM_RAM_ORIGIN, LENGTH = $MEM_RAM_LENGTH
}

/* minimum sizes for heap and stack. It will be checked that they can fit on the RAM */
__stack_size     = $MIN_STACK_SIZE;
__heap_size      = $MIN_HEAP_SIZE;


SECTIONS
{
  /* ================ ROM ================ */

  .text : {
    *(.text .text.* )
  }  > ROM

  .rodata : {
    *(.rodata .rodata.*)
  } > ROM
  .srodata : {
    *(.srodata .srodata.*)
  } > ROM



  /* ================ RAM ================ */

  .init_array : {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(.init_array .init_array.*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > RAM
  .fini_array : {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(.fini_array .fini_array.*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > RAM

  .gcc_except_table : {
    *(.gcc_except_table .gcc_except_table.*)
  } > RAM

  .eh_frame : {
    KEEP (*(.eh_frame))
  } > RAM

  __data_start = .;
  .data : {
      *(.data .data.*)
  } > RAM
  __sdata_start = .;
  .sdata : {
      *(.sdata .sdata.*)
  } > RAM

  __bss_start = .;
  .sbss : {
      *(.sbss .sbss.*)
  } > RAM
  .bss : {
      *(.bss .bss.*)
  } > RAM
  _end = .;

  /* do not place anything after this address, because the heap starts here! */

  /* point the global pointer so it can access sdata, sbss, data and bss */
  __global_pointer$ = MIN(__sdata_start + 0x800, MAX(__data_start + 0x800, _end - 0x800));

  /* stack pointer starts at the top of the ram */
  __stack = ORIGIN(RAM) + LENGTH(RAM);
  .stack : {
    ASSERT ((__stack > (_end + __heap_size + __stack_size)), "Error: RAM too small for heap and stack");
  }
}
EOT
