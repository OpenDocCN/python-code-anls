# `.\pytorch\benchmarks\operator_benchmark\benchmark_all_test.py`

```
# 导入 benchmark_all_other_test 模块，并标记为不会使用 F401 错误，表示它在当前代码中未被直接使用
import benchmark_all_other_test  # noqa: F401

# 导入 benchmark_all_quantized_test 模块，并标记为不会使用 F401 错误，表示它在当前代码中未被直接使用
import benchmark_all_quantized_test  # noqa: F401

# 从 pt 模块导入 unary_test，并标记为不会使用 F401 错误，表示它在当前代码中未被直接使用
from pt import unary_test  # noqa: F401

# 导入 operator_benchmark 模块，作为 op_bench 别名
import operator_benchmark as op_bench

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 调用 operator_benchmark 模块中的 benchmark_runner 的 main 函数，执行基准测试
    op_bench.benchmark_runner.main()
```