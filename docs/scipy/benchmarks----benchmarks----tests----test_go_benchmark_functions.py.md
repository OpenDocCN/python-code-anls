# `D:\src\scipysrc\scipy\benchmarks\benchmarks\tests\test_go_benchmark_functions.py`

```
"""
Unit tests for the global optimization benchmark functions
"""
# 导入所需的库
import numpy as np
# 导入自定义的全局优化基准函数模块
from .. import go_benchmark_functions as gbf
# 导入用于检查函数的成员信息的模块
import inspect

# 定义测试类 TestGoBenchmarkFunctions
class TestGoBenchmarkFunctions:

    # 在每个测试方法运行前执行的设置方法
    def setup_method(self):
        # 获取所有在 go_benchmark_functions 模块中定义的类成员信息
        bench_members = inspect.getmembers(gbf, inspect.isclass)
        # 创建一个字典，将每个基准函数名映射到其对应的类
        self.benchmark_functions = {it[0]: it[1] for it in bench_members if
                                    issubclass(it[1], gbf.Benchmark)}

    # 在每个测试方法运行后执行的清理方法
    def teardown_method(self):
        pass

    # 测试函数，验证函数在给定最优解时返回全局最小值
    def test_optimum_solution(self):
        # 遍历所有基准函数
        for name, klass in self.benchmark_functions.items():
            # 过滤掉 LennardJones 和部分名称以 Problem 开头的函数
            if (name in ['Benchmark', 'LennardJones'] or
                 name.startswith('Problem')):
                continue

            # 创建基准函数对象
            f = klass()

            # 对于 Damavandi 和 Csendes 函数，处理可能的浮点数错误
            if name in ['Damavandi', 'Csendes']:
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 输出函数名、使用全局最优解计算的函数值和全局最优值
                    print(name, f.fun(np.asarray(f.global_optimum[0])),
                          f.fglob)
                    # 验证函数值是否为 NaN
                    assert np.isnan(f.fun(np.asarray(f.global_optimum[0])))
                    continue

            # 输出函数名、使用全局最优解计算的函数值和全局最优值
            print(name, f.fun(np.asarray(f.global_optimum[0])), f.fglob)
            # 验证函数在全局最优解处是否成功
            assert f.success(f.global_optimum[0])

    # 测试函数，验证每个基准函数都有最小能量值
    def test_solution_exists(self):
        # 遍历所有基准函数
        for name, klass in self.benchmark_functions.items():
            # 跳过 Benchmark 函数
            if name == 'Benchmark':
                continue

            # 创建基准函数对象
            f = klass()
            # 获取 f.fglob 属性，如果不存在则引发 AttributeError
            _ = f.fglob

    # 测试函数，验证能够访问基准函数的 bounds 属性
    def test_bounds_access_subscriptable(self):
        # 遍历所有基准函数
        for name, klass in self.benchmark_functions.items():
            # 跳过 Benchmark 和部分名称以 Problem 开头的函数
            if (name == 'Benchmark' or name.startswith('Problem')):
                continue

            # 创建基准函数对象
            f = klass()
            # 访问 f.bounds 的第一个元素
            _ = f.bounds[0]

    # 测试函数，验证问题可以重新调整尺寸
    def test_redimension(self):
        # 获取 LennardJones 基准函数类
        LJ = self.benchmark_functions['LennardJones']
        # 创建 LennardJones 类的实例
        L = LJ()
        # 将问题的维度更改为 10
        L.change_dimensions(10)

        # 检查问题大小变化后，初始向量是否重新调整尺寸
        x0 = L.initial_vector()
        assert len(x0) == 10

        # 现在应该正确长度的边界
        bounds = L.bounds
        assert len(bounds) == 10

        # 检查问题的维度是否为 10
        assert L.N == 10
```