# `.\pytorch\binaries\benchmark_args.h`

```
/**
 * 版权声明：
 * 本代码版权归 Facebook, Inc. 所有
 *
 * 根据 Apache 许可证 2.0 版本（"许可证"）授权；
 * 除非符合许可证要求，否则不得使用本文件。
 * 您可以在以下网址获取许可证副本：
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * 除非适用法律要求或书面同意，否则本软件根据"原样"分发，
 * 无任何明示或暗示的担保或条件。
 * 有关许可证的详细信息，请参阅许可证。
 */
#pragma once

#include "c10/util/Flags.h"

// 定义后端选项，用于指定模型运行时使用的后端
C10_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl, cuda");

// 定义初始化网络的选项，用于指定初始化任何参数的给定网络
C10_DEFINE_string(init_net, "", "The given net to initialize any parameters.");

// 定义输入选项，用于指定网络运行所需的输入数据
C10_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");

// 定义输入维度选项，用于指定输入数据的维度信息
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");

// 定义输入文件选项，用于指定包含输入数据的序列化 protobuf 文件
C10_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");

// 定义输入类型选项，用于指定输入数据的数据类型
C10_DEFINE_string(
    input_type,
    "float",
    "Input type when specifying the input dimension."
    "The supported types are float, uint8_t.");

// 定义迭代次数选项，用于指定运行模型时的迭代次数
C10_DEFINE_int(iter, 10, "The number of iterations to run.");

// 定义内存测量选项，用于指定是否测量加载和运行网络时分配内存的增量
C10_DEFINE_bool(
    measure_memory,
    false,
    "Whether to measure increase in allocated memory while "
    "loading and running the net.");

// 定义网络选项，用于指定要进行基准测试的给定网络
C10_DEFINE_string(net, "", "The given net to benchmark.");

// 定义输出选项，用于指定网络执行完成后应该输出的内容
C10_DEFINE_string(
    output,
    "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");

// 定义输出文件夹选项，用于指定输出内容应写入的文件夹路径
C10_DEFINE_string(
    output_folder,
    "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");

// 定义单独运行选项，用于指定是否对单个运算符进行基准测试
C10_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");

// 定义运行前等待时间选项，用于指定在开始基准测试前等待的秒数
C10_DEFINE_int(
    sleep_before_run,
    0,
    "The seconds to sleep before starting the benchmarking.");

// 定义迭代间等待时间选项，用于指定各个迭代之间等待的秒数
C10_DEFINE_int(
    sleep_between_iteration,
    0,
    "The seconds to sleep between the individual iterations.");

// 定义网络和运算符间等待时间选项，用于指定网络和运算符运行之间等待的秒数
C10_DEFINE_int(
    sleep_between_net_and_operator,
    0,
    "The seconds to sleep between net and operator runs.");

// 定义文本输出选项，用于指定是否为回归测试目的以文本格式输出结果
C10_DEFINE_bool(
    text_output,
    false,
    "Whether to write out output in text format for regression purpose.");
# 定义一个名为 `warmup` 的整数变量，初始值为 0，表示预热迭代次数。
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");

# 定义一个名为 `wipe_cache` 的布尔变量，初始值为 false，表示在运行网络之前是否清空缓存。
C10_DEFINE_bool(
    wipe_cache,
    false,
    "Whether to evict the cache before running network.");
```