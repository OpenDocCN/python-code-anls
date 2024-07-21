# `.\pytorch\benchmarks\operator_benchmark\benchmark_all_other_test.py`

```py
# 导入 `add_test` 到 `tensor_to_test` 的测试模块，用于基准测试
from pt import (
    add_test,                    # 导入加法操作的测试模块
    ao_sparsifier_test,          # 导入 AO 稀疏化器操作的测试模块
    as_strided_test,             # 导入 as_strided 操作的测试模块
    batchnorm_test,              # 导入批归一化操作的测试模块
    binary_test,                 # 导入二进制操作的测试模块
    cat_test,                    # 导入 concatenate 操作的测试模块
    channel_shuffle_test,        # 导入通道混洗操作的测试模块
    chunk_test,                  # 导入分块操作的测试模块
    conv_test,                   # 导入卷积操作的测试模块
    diag_test,                   # 导入对角线操作的测试模块
    embeddingbag_test,           # 导入嵌入包操作的测试模块
    fill_test,                   # 导入填充操作的测试模块
    gather_test,                 # 导入聚集操作的测试模块
    groupnorm_test,              # 导入分组归一化操作的测试模块
    hardsigmoid_test,            # 导入硬 sigmoid 操作的测试模块
    hardswish_test,              # 导入硬 swish 操作的测试模块
    instancenorm_test,           # 导入实例归一化操作的测试模块
    interpolate_test,            # 导入插值操作的测试模块
    layernorm_test,              # 导入层归一化操作的测试模块
    linear_test,                 # 导入线性操作的测试模块
    matmul_test,                 # 导入矩阵乘法操作的测试模块
    nan_to_num_test,             # 导入 NaN 转数字操作的测试模块
    pool_test,                   # 导入池化操作的测试模块
    remainder_test,              # 导入取余操作的测试模块
    softmax_test,                # 导入 softmax 操作的测试模块
    split_test,                  # 导入分割操作的测试模块
    sum_test,                    # 导入求和操作的测试模块
    tensor_to_test,              # 导入张量转换操作的测试模块
)

# 导入操作基准测试模块
import operator_benchmark as op_bench

# 如果这个脚本是作为主程序运行，调用基准测试运行器
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```