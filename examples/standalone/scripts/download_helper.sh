set -e

if [ "$#" -lt 4 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: ./download_helper.sh DEST {gnu|llvm} {2024.09.03|...} {rv32gc_ilp32d|...} [BASENAME [DIST [RELEASE]]]"
    exit 1
fi


dest=$1  # Where to unpack the files
toolchain=$2  # i.e. GCC/LLVM
toolchain_lower=$(echo $toolchain | tr "[:upper:]" "[:lower:]")
version=$3  # i.e. 2023.12.14
lib=$4
basename=${5:-riscv32-unknown-elf}
dist_lower=${6:-$(lsb_release -d | cut -f2 | cut -d' ' -f1 | tr "[:upper:]" "[:lower:]")}
release=${7:-$(lsb_release -r --short)}
ext=${8:-tar.xz}
cpu_arch=$(uname -p)

if [[ "$toolchain_lower" == "llvm" ]]
then
    # TODO: label?
    archive=clang+llvm-$version-$cpu_arch-linux-gnu-$dist_lower-$release.$ext
elif [[ "$toolchain_lower" == "gcc" || "$toolchain_lower" == "gnu" ]]
then
    # TODO: cpu_arch & version
    archive=$basename-$dist_lower-$release-$lib.$ext
    toolchain_lower=gnu
else
    echo "Unsupported toolchain: $toolchain_lower"
    exit 1
fi
URL=https://github.com/PhilippvK/riscv-tools/releases/download/${toolchain_lower}_${version}/$archive
echo "URL=$URL"
wget --progress=dot:giga $URL
mkdir -p $dest
tar -xvf $archive -C $dest
rm $archive