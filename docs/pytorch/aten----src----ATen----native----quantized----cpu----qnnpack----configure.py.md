# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\configure.py`

```py
# 指定 Python 解释器路径，以便系统可以执行该脚本
#!/usr/bin/env python3

# 引入 confu 模块
import confu
# 从 confu 模块中导入 arm 和 x86 子模块
from confu import arm, x86

# 创建标准解析器对象
parser = confu.standard_parser()

# 主函数定义，接受参数 args
def main(args):
    # 使用解析器解析命令行参数
    options = parser.parse_args(args)
    # 根据解析得到的选项构建 Build 对象
    build = confu.Build.from_options(options)

    # 导出头文件 "q8gemm.h" 到目录 "include"
    build.export_cpath("include", ["q8gemm.h"])

    # 进入代码块，设置 build 对象的选项
    with build.options(
        # 指定源代码目录为 "src"
        source_dir="src",
        # 指定依赖的模块列表
        deps=[
            build.deps.cpuinfo,
            build.deps.clog,
            build.deps.psimd,
            build.deps.fxdiv,
            build.deps.pthreadpool,
            build.deps.FP16,
        ],
        # 额外的包含目录 "src"
        extra_include_dirs="src",
    ):

        # 进入代码块，设置 build 对象的选项
        with build.options(
            # 指定源代码目录为 "test"
            source_dir="test",
            # 指定依赖的模块字典
            deps={
                (
                    build,
                    build.deps.cpuinfo,
                    build.deps.clog,
                    build.deps.pthreadpool,
                    build.deps.FP16,
                    build.deps.googletest,
                ): any,
                # 如果目标平台是 Android，将 "log" 设置为真
                "log": build.target.is_android,
            },
            # 额外的包含目录列表 ["src", "test"]
            extra_include_dirs=["src", "test"],
        ):
            # 此处代码块的结束没有直接看到，可能是被省略了，整体结构如上所示
        ):
            # 构建并运行单元测试"hgemm-test"，编译"hgemm.cc"文件
            build.unittest("hgemm-test", build.cxx("hgemm.cc"))
            # 构建并运行单元测试"q8avgpool-test"，编译"q8avgpool.cc"文件
            build.unittest("q8avgpool-test", build.cxx("q8avgpool.cc"))
            # 构建并运行单元测试"q8conv-test"，编译"q8conv.cc"文件
            build.unittest("q8conv-test", build.cxx("q8conv.cc"))
            # 构建并运行单元测试"q8dwconv-test"，编译"q8dwconv.cc"文件
            build.unittest("q8dwconv-test", build.cxx("q8dwconv.cc"))
            # 构建并运行单元测试"q8gavgpool-test"，编译"q8gavgpool.cc"文件
            build.unittest("q8gavgpool-test", build.cxx("q8gavgpool.cc"))
            # 构建并运行单元测试"q8gemm-test"，编译"q8gemm.cc"文件
            build.unittest("q8gemm-test", build.cxx("q8gemm.cc"))
            # 构建并运行单元测试"q8vadd-test"，编译"q8vadd.cc"文件
            build.unittest("q8vadd-test", build.cxx("q8vadd.cc"))
            # 构建并运行单元测试"sconv-test"，编译"sconv.cc"文件
            build.unittest("sconv-test", build.cxx("sconv.cc"))
            # 构建并运行单元测试"sgemm-test"，编译"sgemm.cc"文件
            build.unittest("sgemm-test", build.cxx("sgemm.cc"))
            # 构建并运行单元测试"u8clamp-test"，编译"u8clamp.cc"文件
            build.unittest("u8clamp-test", build.cxx("u8clamp.cc"))
            # 构建并运行单元测试"u8lut32norm-test"，编译"u8lut32norm.cc"文件
            build.unittest("u8lut32norm-test", build.cxx("u8lut32norm.cc"))
            # 构建并运行单元测试"u8maxpool-test"，编译"u8maxpool.cc"文件
            build.unittest("u8maxpool-test", build.cxx("u8maxpool.cc"))
            # 构建并运行单元测试"u8rmax-test"，编译"u8rmax.cc"文件
            build.unittest("u8rmax-test", build.cxx("u8rmax.cc"))
            # 构建并运行单元测试"x8lut-test"，编译"x8lut.cc"文件
            build.unittest("x8lut-test", build.cxx("x8lut.cc"))
            # 构建并运行单元测试"x8zip-test"，编译"x8zip.cc"文件
            build.unittest("x8zip-test", build.cxx("x8zip.cc"))

            # 构建并运行单元测试"add-test"，编译"add.cc"文件
            build.unittest("add-test", build.cxx("add.cc"))
            # 构建并运行单元测试"average-pooling-test"，编译"average-pooling.cc"文件
            build.unittest("average-pooling-test", build.cxx("average-pooling.cc"))
            # 构建并运行单元测试"channel-shuffle-test"，编译"channel-shuffle.cc"文件
            build.unittest("channel-shuffle-test", build.cxx("channel-shuffle.cc"))
            # 构建并运行单元测试"clamp-test"，编译"clamp.cc"文件
            build.unittest("clamp-test", build.cxx("clamp.cc"))
            # 构建并运行单元测试"convolution-test"，编译"convolution.cc"文件
            build.unittest("convolution-test", build.cxx("convolution.cc"))
            # 构建并运行单元测试"deconvolution-test"，编译"deconvolution.cc"文件
            build.unittest("deconvolution-test", build.cxx("deconvolution.cc"))
            # 构建并运行单元测试"fully-connected-test"，编译"fully-connected.cc"文件
            build.unittest("fully-connected-test", build.cxx("fully-connected.cc"))
            # 构建并运行单元测试"global-average-pooling-test"，编译"global-average-pooling.cc"文件
            build.unittest("global-average-pooling-test", build.cxx("global-average-pooling.cc"))
            # 构建并运行单元测试"leaky-relu-test"，编译"leaky-relu.cc"文件
            build.unittest("leaky-relu-test", build.cxx("leaky-relu.cc"))
            # 构建并运行单元测试"max-pooling-test"，编译"max-pooling.cc"文件
            build.unittest("max-pooling-test", build.cxx("max-pooling.cc"))
            # 构建并运行单元测试"sigmoid-test"，编译"sigmoid.cc"文件
            build.unittest("sigmoid-test", build.cxx("sigmoid.cc"))
            # 构建并运行单元测试"softargmax-test"，编译"softargmax.cc"文件
            build.unittest("softargmax-test", build.cxx("softargmax.cc"))
            # 构建并运行单元测试"tanh-test"，编译"tanh.cc"文件
            build.unittest("tanh-test", build.cxx("tanh.cc"))
            # 构建并运行单元测试"hardsigmoid-test"，编译"hardsigmoid.cc"文件
            build.unittest("hardsigmoid-test", build.cxx("hardsigmoid.cc"))
            # 构建并运行单元测试"hardswish-test"，编译"hardswish.cc"文件
            build.unittest("hardswish-test", build.cxx("hardswish.cc"))
            # 构建并运行单元测试"requantization-test"，编译"requantization.cc"文件和相关对象
            build.unittest(
                "requantization-test",
                [build.cxx("requantization.cc")] + requantization_objects,
            )

        # 设置基准测试的ISA（指令集架构）为None
        benchmark_isa = None
        # 如果编译目标是 ARM 架构，设置基准测试的ISA为 ARM NEON
        if build.target.is_arm:
            benchmark_isa = arm.neon
        # 如果编译目标是 x86 架构，设置基准测试的ISA为 x86 SSE4.1
        elif build.target.is_x86:
            benchmark_isa = x86.sse4_1
        # 使用给定的选项来配置基准测试环境
        with build.options(
            source_dir="bench",
            # 指定依赖项，包括构建系统、CPU信息、日志、线程池、FP16和Google Benchmark库
            deps={
                (
                    build,
                    build.deps.cpuinfo,
                    build.deps.clog,
                    build.deps.pthreadpool,
                    build.deps.FP16,
                    build.deps.googlebenchmark,
                ): any,
                # 如果目标平台是 Android，则包含日志依赖
                "log": build.target.is_android,
            },
            # 设置基准测试的ISA（指令集架构）
            isa=benchmark_isa,
            # 添加额外的包含目录路径
            extra_include_dirs="src",
    # 循环调用 build.benchmark() 方法，为各个基准测试添加构建命令
    build.benchmark("add-bench", build.cxx("add.cc"))
    build.benchmark("average-pooling-bench", build.cxx("average-pooling.cc"))
    build.benchmark("channel-shuffle-bench", build.cxx("channel-shuffle.cc"))
    build.benchmark("convolution-bench", build.cxx("convolution.cc"))
    build.benchmark(
        "global-average-pooling-bench", build.cxx("global-average-pooling.cc")
    )
    build.benchmark("max-pooling-bench", build.cxx("max-pooling.cc"))
    build.benchmark("sigmoid-bench", build.cxx("sigmoid.cc"))
    build.benchmark("softargmax-bench", build.cxx("softargmax.cc"))
    build.benchmark("tanh-bench", build.cxx("tanh.cc"))
    build.benchmark("hardsigmoid-bench", build.cxx("hardsigmoid.cc"))
    build.benchmark("hardswish-bench", build.cxx("hardswish.cc"))

    # 添加三个特定的基准测试构建命令
    build.benchmark("q8gemm-bench", build.cxx("q8gemm.cc"))
    build.benchmark("hgemm-bench", build.cxx("hgemm.cc"))
    build.benchmark("sgemm-bench", build.cxx("sgemm.cc"))
    
    # 添加复杂的基准测试构建命令，包括一个需要额外目标的情况
    build.benchmark(
        "requantization-bench",
        [build.cxx("requantization.cc")] + requantization_objects,
    )

return build
# 如果当前脚本作为主程序执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 导入 sys 模块，用于处理命令行参数
    import sys

    # 调用 main 函数并传入命令行参数列表（去除脚本名后的参数）
    main(sys.argv[1:]).generate()
```