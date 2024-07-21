# `.\pytorch\test\jit\test_complexity.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import contextlib  # 上下文管理工具
import os  # 系统操作相关
import sys  # 系统相关
import unittest  # 单元测试框架

import torch  # PyTorch

# 将 test/ 目录下的辅助文件加入到模块搜索路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 从 torch.testing._internal.common_utils 中导入函数和变量
from torch.testing._internal.common_utils import (
    IS_FBCODE,  # 是否运行在 FBCODE 环境下
    run_tests,  # 运行测试函数
    set_default_dtype,  # 设置默认张量类型
    suppress_warnings,  # 抑制警告
)

# 从 torch.testing._internal.jit_metaprogramming_utils 中导入函数
from torch.testing._internal.jit_metaprogramming_utils import (
    get_all_nn_module_tests,  # 获取所有神经网络模块的测试
    get_nn_functional_compiled_fn_and_inputs,  # 获取编译的函数及其输入
    get_nn_mod_test_name,  # 获取神经网络模块的测试名称
    nn_functional_tests,  # 神经网络功能测试
    try_get_nn_module_compiled_mod_and_inputs,  # 尝试获取编译的模块及其输入
)

# 从 torch.testing._internal.jit_utils 中导入函数和类
from torch.testing._internal.jit_utils import (
    enable_profiling_mode,  # 启用性能分析模式
    JitTestCase,  # JIT 测试用例基类
)


# 定义一个函数，用于计算图中的循环和条件语句的数量
def num_ifs_loops(graph):
    graph_str = str(graph)
    # 仅查看图的主体部分
    graph_body = graph_str[0 : graph_str.find("return")]
    return graph_body.count("prim::Loop") + graph_body.count("prim::If")


# 定义一个函数，用于计算基本块中非张量节点的数量
def num_non_tensor_nodes(block):
    num_non_tensor = 0
    for node in block.nodes():
        kind = node.kind()
        # GetAttr 不提供有用的信号，除非在冻结时进行优化
        # Constant 不会被执行，Bailout 应该作为独立的测试处理，这里不提供有用的信号
        if kind == "prim::Constant" or "prim::Bailout" in kind or "GetAttr" in kind:
            continue
        for b in node.blocks():
            num_non_tensor += num_non_tensor_nodes(b)
        tensor_out = False
        for out in node.outputs():
            if "Tensor" in str(out.type()):
                tensor_out = True
                break
        num_non_tensor += int(not tensor_out)
    return num_non_tensor


# 定义一个测试类，继承自 JitTestCase
class TestComplexity(JitTestCase):
    def setUp(self):
        super().setUp()
        self.grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(set_default_dtype(torch.double))

    def tearDown(self):
        self._stack.close()
        torch.set_grad_enabled(self.grad_enabled)
        super().tearDown()

    # 使用 @suppress_warnings 装饰器修饰的测试方法，用于测试生成的功能测试
    @suppress_warnings
    def test_generated_functional_tests(self):
        # 启用性能分析模式
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            # 遍历神经网络功能测试
            for test in nn_functional_tests:
                test_name = test[0]

                # 获取编译的函数及其输入
                fn, inputs = get_nn_functional_compiled_fn_and_inputs(*test)
                # 执行函数多次以填充优化图
                for _ in range(6):
                    fn(*inputs)

                # 获取最近执行的优化图
                g = torch.jit.last_executed_optimized_graph()
                # 计算图中的循环和条件语句数量
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))
        # 打印统计信息
        for line in stats:
            print(line)

    # 使用 @suppress_warnings 装饰器修饰的测试方法，如果运行在 FBCODE 环境下则跳过
    @suppress_warnings
    @unittest.skipIf(IS_FBCODE, "Causes a RecursionError in fbcode")
    # 定义一个测试方法，测试神经网络模块的功能
    def test_nn_module_tests(self):
        # 进入性能分析模式，用于收集测试执行的性能统计信息
        with enable_profiling_mode():
            # 初始化一个统计数据列表，包含表头信息
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            
            # 遍历所有神经网络模块的测试用例
            for test in get_all_nn_module_tests():
                # 尝试获取经过编译的模块和输入数据
                out = try_get_nn_module_compiled_mod_and_inputs(**test)
                
                # 如果获取不到模块和输入数据，继续下一个测试用例
                if not out:
                    continue

                # 解包模块和输入数据
                mod, inputs = out
                # 获取当前测试的名称
                test_name = get_nn_mod_test_name(**test)
                
                # 运行模块多次，以收集性能数据
                for _ in range(6):
                    mod(*inputs)

                # 获取最近执行的优化图
                g = torch.jit.last_executed_optimized_graph()
                
                # 收集当前测试的统计信息：测试名称、条件语句和循环的数量、非张量节点的数量
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))

            # 打印所有收集到的统计信息
            for line in stats:
                print(line)
# 如果这个模块是直接运行的（而不是被导入到其他模块中），执行下面的代码
if __name__ == "__main__":
    # 调用函数运行测试用例
    run_tests()
```