#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 DEST [DBT_REF [BUILD_TYPE]]"
    exit 1
fi

DBT_INSTALL_DIR=$(readlink -f $1)
DBT_SRC_DIR=$(dirname $DBT_INSTALL_DIR)/dbt_src
DBT_REF=${2:-6557dd6bdcda150fc3bb4680fc998e60dc6e51c9}
CMAKE_BUILD_TYPE=${3:-Release}
DBT_BUILD_DIR=$DBT_SRC_DIR/build


NPROC=$(nproc)

if [[ -d $DBT_SRC_DIR ]]
then
    echo "DBT already cloned!"
else
    git clone https://github.com/Minres/DBT-RISE-RISCV.git $DBT_SRC_DIR --recursive
fi

git -C $DBT_SRC_DIR checkout $DBT_REF
git -C $DBT_SRC_DIR submodule update --init --recursive

mkdir -p $DBT_BUILD_DIR

cd $DBT_SRC_DIR
which conan || (echo "Conan not found! Please install with `pip install conan`.")
# conan install . -of build_conan -g CMakeToolchain -g CMakeDeps --build=missing
# conan install .

cmake --preset $CMAKE_BUILD_TYPE -S $DBT_SRC_DIR -B $DBT_BUILD_DIR -DCMAKE_INSTALL_PREFIX=$DBT_INSTALL_DIR

cmake --build $DBT_BUILD_DIR -j$NPROC
cmake --install $DBT_BUILD_DIR
cd -
