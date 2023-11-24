# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: compile script

#!/bin/bash

# 在脚本执行期间更加严格地处理错误，提高脚本的健壮性和可靠性
set -euo pipefail

echo "========== build enter =========="

# 获取脚本所在目录
WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

CUDA_ARCHITECTURE=86 # a: (NVIDIA A100: 80, RTX3080Ti / RTX3090 / RTX A6000: 86)
BUILD_TYPE=Release # t: (Debug, Release)
VERBOSE_MAKEFILE=OFF # b: (ON, OFF)

# 解析命令行参数
while getopts ":a:t:b:" opt
do
    case $opt in
        a)
        CUDA_ARCHITECTURE=$OPTARG
        echo "CUDA_ARCHITECTURE: $CUDA_ARCHITECTURE"
        ;;
        t)
        BUILD_TYPE=$OPTARG
        echo "BUILD_TYPE: $BUILD_TYPE"
        ;;
        b)
        VERBOSE_MAKEFILE=$OPTARG
        echo "VERBOSE_MAKEFILE: $VERBOSE_MAKEFILE"
        ;;
        ?)
        echo "invalid param: $OPTARG"
        exit 1
        ;;
    esac
done

# 执行给定的命令之前，先打印该命令的内容，然后再执行这个命令
echo_cmd() {
    echo $1
    $1
}

echo "========== build flash_attention_inference =========="

echo_cmd "rm -rf build output"
echo_cmd "mkdir build"
echo_cmd "mkdir output"

echo_cmd "cd build"
echo_cmd "cmake -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DFAI_VERBOSE_MAKEFILE=$VERBOSE_MAKEFILE -DCMAKE_INSTALL_PREFIX=$WORK_PATH/output -DCMAKE_SKIP_RPATH=ON .."
echo_cmd "make -j$(nproc --ignore=2)"  # 线程数量：系统可用的逻辑处理器数量减去2
echo_cmd "make install"

echo "========== build info =========="

# 获取当前 Git 仓库中 HEAD 引用（指向当前所在分支）的简写名称
BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse HEAD`
GCC_VERSION=`gcc -dumpversion`
COMPILE_TIME=$(date "+%H:%M:%S %Y-%m-%d")

echo "branch: $BRANCH" >> $WORK_PATH/output/fai_version
echo "commit: $COMMIT" >> $WORK_PATH/output/fai_version
echo "gcc_version: $GCC_VERSION" >> $WORK_PATH/output/fai_version
echo "compile_time: $COMPILE_TIME" >> $WORK_PATH/output/fai_version

echo "========== build exit =========="
