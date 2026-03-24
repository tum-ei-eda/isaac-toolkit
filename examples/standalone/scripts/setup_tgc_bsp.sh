#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters!"
    echo "Usage: $0 DEST [TGC_BSP_REF]"
    exit 1
fi

TGC_BSP_DIR=$(readlink -f $1)
TGC_BSP_REF=${2:-c8ea882b3ef0f5bb476041edd50765801bfea9be}

if [[ -d $TGC_BSP_DIR ]]
then
    echo "TGC BSP already cloned!"
else
    git clone https://git.minres.com/Firmware/MNRS-BM-BSP.git $TGC_BSP_DIR --recursive
fi

git -C $TGC_BSP_DIR checkout $TGC_BSP_REF
