# `.\pytorch\test\fx\test_matcher_utils.py`

```py
# Owner(s): ["module: fx"]

# 导入必要的库和模块
import os
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx

# 获取当前脚本所在目录的父目录，并将其添加到系统路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入单元测试模块
import unittest

# 导入用于图模式匹配的工具类和函数
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)

# 导入测试辅助函数和类
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
from torch.testing._internal.jit_utils import JitTestCase


# 定义一个简单的包装模块，将传入的函数作为其正向传播方法
class WrapperModule(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# 定义一个测试类，继承自 JitTestCase
class TestMatcher(JitTestCase):
    def test_subgraph_matcher_with_attributes(self):
        # 定义一个大模型类，包含权重和偏置的加权求和操作
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._weight = torch.nn.Parameter(torch.ones(3, 3))
                self._bias = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias, x, self._weight)

        # 使用 symbolic_trace 对大模型进行符号化追踪，获取其图表示
        large_model_graph = symbolic_trace(LargeModel()).graph

        # 定义一个模式模型类，具有不同的权重和偏置，同样进行加权求和操作
        class PatternModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
                self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

        # 使用 torch.fx.symbolic_trace 对模式模型进行符号化追踪，获取其图表示
        pattern_graph = torch.fx.symbolic_trace(PatternModel()).graph

        # 创建 SubgraphMatcher 实例，用于在大模型图中查找模式模型图的子图匹配
        subgraph_matcher = SubgraphMatcher(pattern_graph)

        # 对大模型图执行子图匹配，返回匹配结果
        match_result = subgraph_matcher.match(large_model_graph)

        # 断言匹配结果的长度为 1
        self.assertEqual(len(match_result), 1)
    # 定义测试方法，用于测试子图匹配器是否能正确识别包含列表参数的函数
    def test_subgraph_matcher_with_list(self):
        # 定义原始函数 `original`，该函数调用 torch 库的操作符对输入进行视图变换
        def original(x, y):
            return torch.ops.aten.view(x, [5, y.shape[0]])

        # 对 `original` 函数进行符号化跟踪，获取其计算图
        original_graph = torch.fx.symbolic_trace(original).graph

        # 定义模式函数 `pattern`，该函数也调用 torch 库的操作符对输入进行视图变换，但使用参数 `z` 代替了常量 5
        def pattern(x, y, z):
            return torch.ops.aten.view(x, [z, y.shape[0]])

        # 对 `pattern` 函数进行符号化跟踪，获取其计算图
        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        # 创建子图匹配器对象，传入模式图 `pattern_graph`
        subgraph_matcher = SubgraphMatcher(pattern_graph)
        
        # 使用子图匹配器匹配原始图 `original_graph`，返回匹配结果
        match_result = subgraph_matcher.match(original_graph)
        
        # 断言匹配结果的长度为 1，表示找到了一个匹配的子图
        self.assertEqual(len(match_result), 1)

    # 定义测试方法，用于测试子图匹配器是否能正确识别包含错误参数的函数
    def test_subgraph_matcher_with_list_bad(self):
        # 定义原始函数 `original`，该函数调用 torch 库的特定操作符对输入进行重塑，并使用不匹配的参数列表
        def original(x, y):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [1, y.shape[0]], [y.shape[1], y.shape[1]]
            )

        # 对 `original` 函数进行符号化跟踪，获取其计算图
        original_graph = torch.fx.symbolic_trace(original).graph

        # 定义模式函数 `pattern`，该函数也调用 torch 库的特定操作符对输入进行重塑，并使用不匹配的参数列表
        def pattern(x, y, b):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [b, y.shape[0], y.shape[1]], [y.shape[1]]
            )

        # 对 `pattern` 函数进行符号化跟踪，获取其计算图
        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        # 创建子图匹配器对象，传入模式图 `pattern_graph`
        subgraph_matcher = SubgraphMatcher(pattern_graph)
        
        # 使用子图匹配器匹配原始图 `original_graph`，返回匹配结果
        match_result = subgraph_matcher.match(original_graph)
        
        # 断言匹配结果的长度为 0，表示未找到匹配的子图
        self.assertEqual(len(match_result), 0)

    # 定义测试方法，用于测试子图匹配器是否能正确忽略文字字面常量的情况
    def test_subgraph_matcher_ignore_literals(self):
        # 定义原始函数 `original`，该函数对输入矩阵执行加一操作
        def original(x):
            return x + 1

        # 利用 `make_fx` 函数创建 `original` 函数的计算图，并进行死代码消除
        original_graph = make_fx(original)(torch.ones(3, 3)).graph
        original_graph.eliminate_dead_code()

        # 定义模式函数 `pattern`，该函数对输入矩阵执行加二操作
        def pattern(x):
            return x + 2

        # 利用 `make_fx` 函数创建 `pattern` 函数的计算图，并进行死代码消除
        pattern_graph = make_fx(pattern)(torch.ones(4, 4)).graph
        pattern_graph.eliminate_dead_code()

        # 创建子图匹配器对象，传入模式图 `pattern_graph`，并设置忽略文字字面常量为 True
        subgraph_matcher = SubgraphMatcher(pattern_graph, ignore_literals=True)
        
        # 使用子图匹配器匹配原始图 `original_graph`，返回匹配结果
        match_result = subgraph_matcher.match(original_graph)
        
        # 断言匹配结果的长度为 1，表示找到了一个匹配的子图（忽略了字面常量）
        self.assertEqual(len(match_result), 1)
    def test_variatic_arg_matching(self):
        # 准备输入数据，一个元组，包含一个随机生成的张量
        inputs = (torch.randn(20, 16, 50, 32),)

        # 定义一个 maxpool 函数，使用 torch.ops.aten.max_pool2d_with_indices.default 函数进行池化操作
        def maxpool(x, kernel_size, stride, padding, dilation):
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding, dilation
            )

        # 对 maxpool 函数进行符号化追踪，得到其图表达式
        maxpool_graph = torch.fx.symbolic_trace(maxpool).graph

        # 创建一个 SubgraphMatcher 对象，用于匹配 maxpool_graph
        maxpool_matcher = SubgraphMatcher(maxpool_graph)
        # 在 maxpool_graph 上进行匹配操作
        match_result = maxpool_matcher.match(maxpool_graph)
        # 断言匹配结果的长度为 1
        self.assertEqual(len(match_result), 1)

        # 创建一个 MaxPool2d 模块，只包含 "stride" 参数的图表达式
        maxpool_s = torch.nn.MaxPool2d(kernel_size=2, stride=1).eval()
        # 对 maxpool_s 模块进行符号化执行，得到其图表达式
        maxpool_s_graph = make_fx(maxpool_s)(*inputs).graph
        # 在 maxpool_graph 上进行匹配操作
        match_s_result = maxpool_matcher.match(maxpool_s_graph)
        # 断言匹配结果的长度为 1
        self.assertEqual(len(match_s_result), 1)

        # 创建一个 MaxPool2d 模块，只包含 "padding" 参数的图表达式
        maxpool_p = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        # 对 maxpool_p 模块进行符号化执行，得到其图表达式
        maxpool_p_graph = make_fx(maxpool_p)(*inputs).graph
        # 在 maxpool_graph 上进行匹配操作
        match_p_result = maxpool_matcher.match(maxpool_p_graph)
        # 断言匹配结果的长度为 1
        self.assertEqual(len(match_p_result), 1)

        # 创建一个 MaxPool2d 模块，包含 "stride, padding" 参数的图表达式
        maxpool_sp = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        # 对 maxpool_sp 模块进行符号化执行，得到其图表达式
        maxpool_sp_graph = make_fx(maxpool_sp)(*inputs).graph
        # 在 maxpool_graph 上进行匹配操作
        match_sp_result = maxpool_matcher.match(maxpool_sp_graph)
        # 断言匹配结果的长度为 1
        self.assertEqual(len(match_sp_result), 1)

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_split_to_graph_and_name_node_map(self):
        """Testing the internal helper function for splitting the pattern graph"""
        # 导入内部的辅助函数 _split_to_graph_and_name_node_map
        from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
            _split_to_graph_and_name_node_map,
        )

        # 定义一个 pattern 函数，对输入进行卷积、ReLU 激活和乘以 2 的操作，并返回相应的结果和字典
        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        # 导入 capture_pre_autograd_graph 函数
        from torch._export import capture_pre_autograd_graph

        # 准备示例输入
        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        # 对 pattern 函数进行前向图捕获，得到图模块
        pattern_gm = capture_pre_autograd_graph(WrapperModule(pattern), example_inputs)
        # 对示例输入执行 pattern_gm 模块
        before_split_res = pattern_gm(*example_inputs)
        # 使用 _split_to_graph_and_name_node_map 函数，将 pattern_gm 拆分为图模块和名称节点映射
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        # 对示例输入再次执行 pattern_gm 模块
        after_split_res = pattern_gm(*example_inputs)
        # 断言拆分前后的结果一致性
        self.assertEqual(before_split_res[0], after_split_res[0])
        self.assertEqual(before_split_res[1], after_split_res[1])

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    # 定义测试函数，用于测试带有名称节点映射的子图匹配器，使用函数模式
    def test_matcher_with_name_node_map_function(self):
        """Testing SubgraphMatcherWithNameNodeMap with function pattern"""

        # 定义目标图函数，接受输入 x 和 weight，并进行卷积操作
        def target_graph(x, weight):
            # 将输入 x 增加两倍
            x = x * 2
            # 将权重 weight 增加三倍
            weight = weight * 3
            # 执行二维卷积操作
            conv = F.conv2d(x, weight)
            # 执行 ReLU 激活函数
            relu = F.relu(conv)
            # 将 ReLU 结果乘以2
            relu2 = relu * 2
            # 返回 ReLU 结果与乘以2的结果的和
            return relu + relu2

        # 定义模式图函数，接受输入 x 和 weight，并执行卷积操作，返回 ReLU 结果、乘以2的 ReLU 结果和节点名称到节点映射的字典
        def pattern(x, weight):
            # 执行二维卷积操作
            conv = F.conv2d(x, weight)
            # 执行 ReLU 激活函数
            relu = F.relu(conv)
            # 将 ReLU 结果乘以2
            relu_mul_by_two = relu * 2
            # 返回 ReLU 结果、乘以2的 ReLU 结果和节点名称到节点映射的字典
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        # 导入捕获预自动求图函数
        from torch._export import capture_pre_autograd_graph

        # 准备示例输入
        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        # 使用 WrapperModule 包装模式图函数，并捕获其预自动求图
        pattern_gm = capture_pre_autograd_graph(WrapperModule(pattern), example_inputs)
        # 创建子图匹配器，使用捕获的模式图函数的预自动求图
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        # 使用 WrapperModule 包装目标图函数，并捕获其预自动求图
        target_gm = capture_pre_autograd_graph(
            WrapperModule(target_graph), example_inputs
        )
        # 匹配目标图中的子图，并返回内部匹配结果列表
        internal_matches = matcher.match(target_gm.graph)
        # 遍历内部匹配结果列表
        for internal_match in internal_matches:
            # 获取节点名称到节点映射的字典
            name_node_map = internal_match.name_node_map
            # 断言节点映射中包含 "conv" 和 "relu"
            assert "conv" in name_node_map
            assert "relu" in name_node_map
            # 给节点映射中的 "conv" 节点的元数据添加自定义注释
            name_node_map["conv"].meta["custom_annotation"] = "annotation"
            # 检查是否正确注释了目标图模块
            for n in target_gm.graph.nodes:
                if n == name_node_map["conv"]:
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )
    def test_matcher_with_name_node_map_module(self):
        """Testing SubgraphMatcherWithNameNodeMap with module pattern"""
        
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
    
            def forward(self, x):
                return self.linear(x)
    
        class Pattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
    
            def forward(self, x):
                # 计算线性层的输出
                linear = self.linear(x)
                # 返回线性层输出和包含自定义注释的字典
                # 注意: 由于 nn.Parameter 不是 Dynamo 中允许的输出类型，所以不能将 "weight": self.linear.weight 放入字典中
                return linear, {"linear": linear, "x": x}
    
        from torch._export import capture_pre_autograd_graph
    
        example_inputs = (torch.randn(3, 5),)
        # 捕获 Pattern 模块的前自动微分图
        pattern_gm = capture_pre_autograd_graph(Pattern(), example_inputs)
        # 使用捕获的 Pattern 图创建 SubgraphMatcherWithNameNodeMap 对象
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        # 捕获 M 模块的前自动微分图
        target_gm = capture_pre_autograd_graph(M(), example_inputs)
        # 使用 matcher 匹配目标图中的子图
        internal_matches = matcher.match(target_gm.graph)
        # 遍历所有匹配到的内部匹配
        for internal_match in internal_matches:
            # 获取内部匹配的名称节点映射
            name_node_map = internal_match.name_node_map
            # 断言 "linear" 在名称节点映射中
            assert "linear" in name_node_map
            # 断言 "x" 在名称节点映射中
            assert "x" in name_node_map
            # 向名称节点映射中的 "linear" 节点的元数据添加自定义注释
            name_node_map["linear"].meta["custom_annotation"] = "annotation"
            # 检查是否正确地对目标图模块进行了注释
            for n in target_gm.graph.nodes:
                if n == name_node_map["linear"]:
                    # 断言已添加 "custom_annotation" 到节点元数据，并且其值为 "annotation"
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```