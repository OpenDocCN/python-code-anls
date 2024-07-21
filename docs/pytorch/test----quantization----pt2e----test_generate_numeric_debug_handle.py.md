# `.\pytorch\test\quantization\pt2e\test_generate_numeric_debug_handle.py`

```py
# Owner(s): ["oncall: quantization"]

import unittest  # 引入 unittest 模块

import torch  # 引入 PyTorch 库
from torch._export import capture_pre_autograd_graph  # 导入捕获自动求导图的函数
from torch.ao.quantization import generate_numeric_debug_handle  # 导入生成数值调试句柄的函数
from torch.ao.quantization.pt2e.export_utils import _WrapperModule  # 导入包装模块类
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # 导入量化相关函数
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,  # 导入对称量化配置函数
    XNNPACKQuantizer,  # 导入 XNNPACK 量化器类
)
from torch.fx import Node  # 导入 Torch FX 的节点类
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,  # 导入带有名称节点映射的子图匹配工具类
)
from torch.testing._internal.common_quantization import TestHelperModules  # 导入测试辅助模块类
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase  # 导入测试工具和测试用例基类


def _extract_conv2d_pattern_debug_handle_map(model):
    """Returns a debug_handle_map from input/weight/bias/output to numeric_debug_handle
    for conv2d pattern, extracted from the model
    """

    def conv_pattern(input, weight, bias):
        output = torch.nn.functional.conv2d(input, weight, bias)  # 执行卷积操作
        return output, {
            "input": input,   # 将输入数据放入字典中
            "weight": weight,  # 将权重放入字典中
            "bias": bias,      # 将偏置放入字典中
            "output": output,  # 将输出数据放入字典中
        }

    conv_pattern_example_inputs = (
        torch.randn(1, 1, 3, 3),  # 生成一个随机的输入张量
        torch.randn(1, 1, 1, 1),  # 生成一个随机的权重张量
        torch.randn(1),           # 生成一个随机的偏置张量
    )
    conv_gm = capture_pre_autograd_graph(
        _WrapperModule(conv_pattern), conv_pattern_example_inputs  # 捕获卷积模式的预自动求导图
    )
    conv_pm = SubgraphMatcherWithNameNodeMap(conv_gm)  # 使用名称节点映射的子图匹配器创建子图匹配对象
    matches = conv_pm.match(model.graph)  # 在模型的计算图中进行匹配
    assert len(matches) == 1, "Expecting to have one match"  # 断言确保只有一个匹配结果
    match = matches[0]  # 获取匹配的结果
    name_node_map = match.name_node_map  # 获取名称到节点映射

    input_node = name_node_map["input"]    # 获取输入节点
    weight_node = name_node_map["weight"]  # 获取权重节点
    bias_node = name_node_map["bias"]      # 获取偏置节点
    output_node = name_node_map["output"]  # 获取输出节点

    debug_handle_map = {}  # 初始化调试句柄映射字典
    conv_node = output_node

    # 检查输入节点是否在卷积节点的数值调试句柄元数据中
    if input_node not in conv_node.meta["numeric_debug_handle"]:
        return debug_handle_map  # 如果输入节点不在句柄中，则返回空的调试句柄映射字典

    # 将数值调试句柄映射到调试句柄映射字典中
    debug_handle_map["input"] = conv_node.meta["numeric_debug_handle"][input_node]
    debug_handle_map["weight"] = conv_node.meta["numeric_debug_handle"][weight_node]

    # 如果有偏置节点，则将偏置节点的数值调试句柄映射到调试句柄映射字典中
    if bias_node is not None:
        debug_handle_map["bias"] = conv_node.meta["numeric_debug_handle"][bias_node]

    # 将输出节点的数值调试句柄映射到调试句柄映射字典中
    debug_handle_map["output"] = conv_node.meta["numeric_debug_handle"]["output"]
    return debug_handle_map  # 返回调试句柄映射字典


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestGenerateNumericDebugHandle(TestCase):
    """Unit tests for the `generate_numeric_debug_handle` function."""
    # 定义单元测试方法 test_simple，用于测试 Conv2dThenConv1d 类的功能
    def test_simple(self):
        # 创建 Conv2dThenConv1d 类的实例 m
        m = TestHelperModules.Conv2dThenConv1d()
        # 获取示例输入数据
        example_inputs = m.example_inputs()
        # 捕获 autograd 图前的计算图，返回处理后的模型 m
        m = capture_pre_autograd_graph(m, example_inputs)
        # 生成数值调试句柄，将其关联到模型 m 上
        generate_numeric_debug_handle(m)
        # 初始化一个空集合用于存储唯一的数值调试句柄 ID
        unique_ids = set()
        # 初始化计数器 count
        count = 0
        # 遍历计算图 m 中的节点
        for n in m.graph.nodes:
            # 检查节点的元数据中是否包含 "numeric_debug_handle"
            if "numeric_debug_handle" in n.meta:
                # 遍历节点的参数
                for arg in n.args:
                    # 如果参数是 Node 类型的实例，将其对应的调试句柄 ID 添加到 unique_ids 集合中
                    if isinstance(arg, Node):
                        unique_ids.add(n.meta["numeric_debug_handle"][arg])
                        count += 1
                # 将节点输出对应的调试句柄 ID 添加到 unique_ids 集合中
                unique_ids.add(n.meta["numeric_debug_handle"]["output"])
                count += 1
        # 断言唯一的调试句柄 ID 数量与计数器 count 的值相等
        self.assertEqual(len(unique_ids), count)

    # 定义测试量化转换函数 preserve_handle 的单元测试方法
    def test_quantize_pt2e_preserve_handle(self):
        # 创建 Conv2dThenConv1d 类的实例 m
        m = TestHelperModules.Conv2dThenConv1d()
        # 获取示例输入数据
        example_inputs = m.example_inputs()
        # 捕获 autograd 图前的计算图，返回处理后的模型 m
        m = capture_pre_autograd_graph(m, example_inputs)
        # 生成数值调试句柄，将其关联到模型 m 上
        generate_numeric_debug_handle(m)

        # 提取模型 m 中 Conv2d 模式的调试句柄映射
        debug_handle_map_ref = _extract_conv2d_pattern_debug_handle_map(m)

        # 创建 XNNPACKQuantizer 实例 quantizer，配置为非通道级对称量化
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        # 准备模型 m 的 PyTorch 转换过程，使用 quantizer 进行量化
        m = prepare_pt2e(m, quantizer)
        # 提取量化后模型 m 的 Conv2d 模式的调试句柄映射
        debug_handle_map = _extract_conv2d_pattern_debug_handle_map(m)
        # 断言量化前后的调试句柄映射是否相等
        self.assertEqual(debug_handle_map, debug_handle_map_ref)
        
        # 调用模型 m 处理示例输入数据
        m(*example_inputs)
        # 将模型 m 转换为 PT2E 格式
        m = convert_pt2e(m)
        # 提取转换后模型 m 的 Conv2d 模式的调试句柄映射
        debug_handle_map = _extract_conv2d_pattern_debug_handle_map(m)
        # 断言转换前后的调试句柄映射是否相等
        self.assertEqual(debug_handle_map, debug_handle_map_ref)
```