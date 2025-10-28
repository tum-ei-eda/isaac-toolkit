#!/bin/bash

set -e

if [ "$#" -lt 3 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 DEST {gnu|llvm} {2024.09.03|...} {rv32gc_ilp32d|...} [BASENAME [DIST [RELEASE]]]"
    exit 1
fi


dest=$1  # Where to unpack the files
tool=$2  # i.e. GCC/LLVM
tool_lower=$(echo $tool | tr "[:upper:]" "[:lower:]")
version=$3  # i.e. 2023.12.14
lib=${4:-default}
basename=${5:-riscv32-unknown-elf}
dist_lower=${6:-$(lsb_release -d | cut -f2 | cut -d' ' -f1 | tr "[:upper:]" "[:lower:]")}
release=${7:-$(lsb_release -r --short)}
ext=${8:-tar.xz}
cpu_arch=$(uname -p)


if [[ "$dist_lower" == "pop!_os" ]]
then
    dist_lower=ubuntu
fi

if [[ "$dist_lower" != "ubuntu" ]]
then
    echo "Unsupported distribution: $dist_lower (Currently only Ubuntu/PopOS is supported)".

fi

if [[ "$tool_lower" == "llvm" ]]
then
    # TODO: label?
    archive=clang+llvm-$version-$cpu_arch-linux-gnu-$dist_lower-$release.$ext
elif [[ "$tool_lower" == "gcc" || "$tool_lower" == "gnu" ]]
then
    # TODO: cpu_arch & version
    archive=$basename-$dist_lower-$release-$lib.$ext
    tool_lower=gnu
elif [[ "$tool_lower" == "spike" ]]
then
    # TODO: label?
    archive=spike-x86_64-linux-gnu-$dist_lower-$release.$ext
elif [[ "$tool_lower" == "etiss" ]]
then
    # TODO: label?
    archive=etiss-x86_64-linux-gnu-$dist_lower-$release.$ext
else
    echo "Unsupported tool: $toolchain_lower"
    exit 1
fi
URL=https://github.com/PhilippvK/riscv-tools/releases/download/${tool_lower}_${version}/$archive
echo "URL=$URL"
wget --progress=dot:giga $URL
mkdir -p $dest
tar -xvf $archive -C $dest
rm $archive
