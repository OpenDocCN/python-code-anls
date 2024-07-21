# `.\pytorch\benchmarks\operator_benchmark\benchmark_all_quantized_test.py`

```py
# 导入 pt 模块中的测试函数，忽略 F401 类型的未使用警告
from pt import (
    qactivation_test,          # 激活函数量化测试
    qarithmetic_test,          # 算术操作量化测试
    qatembedding_ops_test,     # AT 嵌入操作量化测试
    qbatchnorm_test,           # 批归一化量化测试
    qcat_test,                 # 拼接操作量化测试
    qcomparators_test,         # 比较操作量化测试
    qconv_test,                # 卷积操作量化测试
    qembedding_pack_test,      # 嵌入打包操作量化测试
    qembeddingbag_test,        # 嵌入袋操作量化测试
    qgroupnorm_test,           # 分组归一化量化测试
    qinstancenorm_test,        # 实例归一化量化测试
    qinterpolate_test,         # 插值操作量化测试
    qlayernorm_test,           # 层归一化量化测试
    qlinear_test,              # 线性操作量化测试
    qobserver_test,            # 观察者量化测试
    qpool_test,                # 池化操作量化测试
    qrnn_test,                 # 循环神经网络量化测试
    qtensor_method_test,       # 张量方法量化测试
    quantization_test,         # 量化测试
    qunary_test,               # 一元操作量化测试
)

# 导入 operator_benchmark 模块并重命名为 op_bench
import operator_benchmark as op_bench

# 如果当前脚本作为主程序运行，则调用 op_bench 模块中的 benchmark_runner.main() 函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```