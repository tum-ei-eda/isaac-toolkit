#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 DEST [TGC_REF [BUILD_TYPE]]"
    exit 1
fi

TGC_INSTALL_DIR=$(readlink -f $1)
TGC_SRC_DIR=$(dirname $TGC_INSTALL_DIR)/tgc_src
TGC_REF=${2:-f55fc06867a11e5f7ab57772167109be9e6c4446}
CMAKE_BUILD_TYPE=${3:-Release}
TGC_BUILD_DIR=$TGC_SRC_DIR/build


NPROC=$(nproc)

if [[ -d $TGC_SRC_DIR ]]
then
    echo "TGC already cloned!"
else
    git clone https://git.minres.com/TGFS/TGC-ISS.git $TGC_SRC_DIR --recursive
fi

git -C $TGC_SRC_DIR checkout $TGC_REF

mkdir -p $TGC_BUILD_DIR

cd $TGC_SRC_DIR
which conan || (echo "Conan not found! Please install with `pip install conan`.")
# conan install . -of build_conan -g CMakeToolchain -g CMakeDeps --build=missing
# conan install .

# cmake -B $TGC_BUILD_DIR -S $TGC_SRC_DIR -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_INSTALL_PREFIX:PATH=$TGC_INSTALL_DIR -DCMAKE_TOOLCHAIN_FILE=$TGC_SRC_DIR/build_conan/build/Release/generators/conan_toolchain.cmake
cmake --preset $CMAKE_BUILD_TYPE -S $TGC_SRC_DIR -B $TGC_BUILD_DIR -DCMAKE_INSTALL_PREFIX=$TGC_INSTALL_DIR

cmake --build $TGC_BUILD_DIR -j$NPROC
cmake --install $TGC_BUILD_DIR
cd -
