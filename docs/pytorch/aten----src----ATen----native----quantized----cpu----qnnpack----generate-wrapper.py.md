# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\generate-wrapper.py`

```py
# 导入标准库中的 os 模块，用于访问操作系统功能
import os

# 定义一个字典 QNNPACK_SOURCES，包含不同平台下的特定源文件列表

QNNPACK_SOURCES = {
    # 通用函数，无特定平台要求
    None: [
        "requantization/fp32-psimd.c",         # 浮点32位重量化的PSIMD实现
        "requantization/fp32-scalar.c",        # 浮点32位重量化的标量实现
        "requantization/gemmlowp-scalar.c",    # Gemmlowp算法的标量实现
        "requantization/precise-psimd.c",      # 精确PSIMD实现
        "requantization/precise-scalar.c",     # 精确标量实现
        "requantization/q31-scalar.c",         # Q31格式的标量实现
        "sgemm/6x8-psimd.c",                    # 6x8矩阵乘法的PSIMD实现
        "u8lut32norm/scalar.c",                 # 无符号8位LUT32规范化的标量实现
        "x8lut/scalar.c",                       # 8位LUT的标量实现
    ],

    # ARM平台特定的uKernels，包括ARM32和ARM64
    "defined(__arm__) || defined(__aarch64__)": [
        "q8avgpool/mp8x9p8q-neon.c",            # Q8均值池化的NEON实现
        "q8avgpool/up8x9-neon.c",               # 无符号Q8均值池化的NEON实现
        "q8avgpool/up8xm-neon.c",               # 无符号Q8多维均值池化的NEON实现
        "q8conv/4x8-neon.c",                    # Q8卷积的4x8 NEON实现
        "q8conv/8x8-neon.c",                    # Q8卷积的8x8 NEON实现
        "q8dwconv/mp8x25-neon.c",               # Q8深度卷积的mp8x25 NEON实现
        "q8dwconv/mp8x25-neon-per-channel.c",   # Q8深度卷积的mp8x25 NEON按通道实现
        "q8dwconv/mp8x27-neon.c",               # Q8深度卷积的mp8x27 NEON实现
        "q8dwconv/up8x9-neon.c",                # 无符号Q8深度卷积的NEON实现
        "q8dwconv/up8x9-neon-per-channel.c",    # 无符号Q8深度卷积的NEON按通道实现
        "q8gavgpool/mp8x7p7q-neon.c",           # Q8全局均值池化的mp8x7p7q NEON实现
        "q8gavgpool/up8x7-neon.c",              # 无符号Q8全局均值池化的NEON实现
        "q8gavgpool/up8xm-neon.c",              # 无符号Q8全局多维均值池化的NEON实现
        "q8gemm/4x-sumrows-neon.c",             # 4x矩阵乘法行求和的NEON实现
        "q8gemm/4x8-neon.c",                    # 4x8矩阵乘法的NEON实现
        "q8gemm/4x8-dq-neon.c",                 # 双量化的4x8矩阵乘法的NEON实现
        "q8gemm/4x8c2-xzp-neon.c",              # 2通道零点扩展的4x8矩阵乘法的NEON实现
        "q8gemm/6x4-neon.c",                    # 6x4矩阵乘法的NEON实现
        "q8gemm/8x8-neon.c",                    # 8x8矩阵乘法的NEON实现
        "q8vadd/neon.c",                        # Q8向量加法的NEON实现
        "requantization/fp32-neon.c",           # 浮点32位重量化的NEON实现
        "requantization/gemmlowp-neon.c",       # Gemmlowp算法的NEON实现
        "requantization/precise-neon.c",        # 精确实现的NEON实现
        "requantization/q31-neon.c",            # Q31格式的NEON实现
        "sgemm/5x8-neon.c",                     # 5x8矩阵乘法的NEON实现
        "sgemm/6x8-neon.c",                     # 6x8矩阵乘法的NEON实现
        "u8clamp/neon.c",                       # 无符号8位数据的NEON实现
        "u8maxpool/16x9p8q-neon.c",              # 16x9 Q8最大池化的NEON实现
        "u8maxpool/sub16-neon.c",               # 无符号16位数据最大池化的NEON实现
        "u8rmax/neon.c",                        # 无符号8位数据最大值的NEON实现
        "x8zip/x2-neon.c",                      # 2维数据压缩的NEON实现
        "x8zip/x3-neon.c",                      # 3维数据压缩的NEON实现
        "x8zip/x4-neon.c",                      # 4维数据压缩的NEON实现
        "x8zip/xm-neon.c",                      # 多维数据压缩的NEON实现
    ],

    # x86/x86-64平台特定的uKernels
    "defined(__i386__) || defined(__i686__) || defined(__x86_64__)": [
        "q8avgpool/mp8x9p8q-sse2.c",            # Q8均值池化的SSE2实现
        "q8avgpool/up8x9-sse2.c",               # 无符号Q8均值池化的SSE2实现
        "q8avgpool/up8xm-sse2.c",               # 无符号Q8多维均值池化的SSE2实现
        "q8conv/4x4c2-sse2.c",                  # Q8卷积的4x4c2 SSE2实现
        "q8dwconv/mp8x25-sse2.c",               # Q8深度卷积的mp8x25 SSE2实现
        "q8dwconv/mp8x25-sse2-per-channel.c",   # Q8深度卷积的mp8x25 SSE2按通道实现
        "q8dwconv/mp8x27-sse2.c",               # Q8深度卷积的mp8x27 SSE2实现
        "q8dwconv/up8x9-sse2.c",                # 无符号Q8深度卷积的SSE2实现
        "q8dwconv/up8x9-sse2-per-channel.c",    # 无符号Q8深度卷积的SSE2按通道实现
        "q8gavgpool/mp8x7p7q-sse2.c",           # Q8全局均值池化的mp8x7p7q SSE2实现
        "q8gavgpool/up8x7-sse2.c",              # 无符号Q8全局均值池化的SSE2实现
        "q8gavgpool/up8xm-sse2.c",              # 无符号Q8全局多维均值池化的SSE2实现
        "q8gemm/2x4c8-sse2.c",                  # 2x4矩阵乘法的4c8 SSE2实现
        "q8gemm/4x4c2-dq-sse2.c",               # 双
    # 如果目标平台为 ARM 架构 (__arm__)，加载以下汇编文件
    "defined(__arm__)": [
        "hgemm/8x8-aarch32-neonfp16arith.S",  # ARM 32位架构下的矩阵乘法优化程序
        "q8conv/4x8-aarch32-neon.S",           # ARM 32位架构下的量化卷积优化程序
        "q8dwconv/up8x9-aarch32-neon.S",       # ARM 32位架构下的深度可分离量化卷积优化程序
        "q8dwconv/up8x9-aarch32-neon-per-channel.S",  # ARM 32位架构下的通道间深度可分离量化卷积优化程序
        "q8gemm/4x8-aarch32-neon.S",           # ARM 32位架构下的量化乘法优化程序
        "q8gemm/4x8-dq-aarch32-neon.S",        # ARM 32位架构下的量化乘法（带饱和运算）优化程序
        "q8gemm/4x8c2-xzp-aarch32-neon.S",     # ARM 32位架构下的量化乘法（带二值化参数）优化程序
    ],
    # 如果目标平台为 AArch64 架构 (__aarch64__)，加载以下汇编文件
    "defined(__aarch64__)": [
        "q8conv/8x8-aarch64-neon.S",           # AArch64 架构下的量化卷积优化程序
        "q8gemm/8x8-aarch64-neon.S",           # AArch64 架构下的量化乘法优化程序
        "q8gemm/8x8-dq-aarch64-neon.S",        # AArch64 架构下的量化乘法（带饱和运算）优化程序
    ],
}

# 定义一个用于生成注释的横幅字符串常量
BANNER = "/* Auto-generated by generate-wrappers.py script. Do not modify */"

# 如果这个脚本作为主程序执行
if __name__ == "__main__":
    # 遍历 QNNPACK_SOURCES 字典中的条件和文件名列表
    for condition, filenames in QNNPACK_SOURCES.items():
        # 遍历每个文件名
        for filename in filenames:
            # 构建文件的完整路径
            filepath = os.path.join("wrappers", filename)
            # 检查文件路径所在的目录是否存在，若不存在则创建
            if not os.path.isdir(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            # 打开文件以写入模式
            with open(filepath, "w") as wrapper:
                # 将横幅字符串写入文件
                print(BANNER, file=wrapper)
                # 写入一个空行
                print(file=wrapper)

                # 如果存在条件，则在文件中写入预处理指令和包含特定文件的语句
                if condition is None:
                    print(f"#include <{filename}>", file=wrapper)
                else:
                    # 只有当条件满足时才包含源文件
                    print(f"#if {condition}", file=wrapper)
                    print(f"#include <{filename}>", file=wrapper)
                    print(f"#endif /* {condition} */", file=wrapper)
```