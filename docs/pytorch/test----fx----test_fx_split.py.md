# `.\pytorch\test\fx\test_fx_split.py`

```py
# Owner(s): ["module: fx"]

# 引入必要的库和模块
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.fx.passes.split_utils import split_by_tags
from torch.testing._internal.common_utils import TestCase

# 定义一个测试类 TestFXSplit，继承自 TestCase
class TestFXSplit(TestCase):
    
    # 定义测试方法 test_split_preserve_node_meta
    def test_split_preserve_node_meta(self):
        
        # 定义一个简单的测试模块 TestModule
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x = x + x  # 计算输入 x 的加法操作
                y = y * y  # 计算输入 y 的乘法操作
                return x - y  # 返回 x 和 y 的差值
        
        # 对 TestModule 进行符号跟踪，得到图模型 gm
        gm = torch.fx.symbolic_trace(TestModule())
        
        # 遍历 gm 图中的每个节点，为每个节点添加元数据
        for node in gm.graph.nodes:
            node.meta["name"] = node.name  # 将节点名称存入节点的元数据中
            # 根据节点名称为节点添加标签
            if node.name == "add":
                node.tag = "a"
            elif node.name == "mul":
                node.tag = "b"
            elif node.name == "sub":
                node.tag = "c"
        
        # 根据标签将 gm 图拆分为多个子图，存入 split_gm 中
        split_gm = split_by_tags(gm, ["a", "b", "c"])
        
        # 遍历每个子图中的节点，并验证节点的元数据被正确复制
        for m in split_gm.children():
            for n in m.graph.nodes:
                if n.op != "output":
                    self.assertIn("name", n.meta)  # 断言节点的元数据中包含名称
                    self.assertEqual(n.meta["name"], n.name)  # 断言节点的元数据名称与节点名称相同
        
        # 验证占位符节点的元数据是否正确复制
        for node in split_gm.graph.nodes:
            if node.op == "placeholder":
                self.assertIn("name", node.meta)  # 断言节点的元数据中包含名称
                self.assertEqual(node.meta["name"], node.name)  # 断言节点的元数据名称与节点名称相同


# 定义一个测试类 TestSplitByTags，继承自 TestCase
class TestSplitByTags(TestCase):
    
    # 定义一个简单的测试模块 TestModule，继承自 torch.nn.Module
    class TestModule(torch.nn.Module):
        
        # 构造方法，初始化线性层
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 3)
            self.linear2 = torch.nn.Linear(4, 5)
            self.linear3 = torch.nn.Linear(6, 7)
            self.linear4 = torch.nn.Linear(8, 6)
        
        # 前向传播方法，接收三个张量输入，返回一个张量
        def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ) -> torch.Tensor:
            v1 = self.linear1(x1)  # 第一个线性层的前向传播
            v2 = self.linear2(x2)  # 第二个线性层的前向传播
            v3 = self.linear3(x3)  # 第三个线性层的前向传播
            v4 = torch.cat([v1, v2, v3])  # 将前三个张量拼接成一个张量 v4
            return self.linear4(v4)  # 对拼接后的张量应用第四个线性层的前向传播
    
    # 静态方法 trace_and_tag，接收一个模块和标签列表作为参数
    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, tags: List[str]
        # 该方法的具体实现将在后续代码中给出，注释此处只声明了方法签名
    ) -> Tuple[torch.fx.GraphModule, Dict[str, List[str]]]:
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            linear1 - tag: "red"
            linear2 - tag: "blue"
            linear3, linear4 - tag: "green"

        At the beginning we have:
            gm:
                linear1
                linear2
                linear3
                linear4

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    linear1
                blue:
                    linear2
                green:
                    linear3
                    linear4
        """
        tag_node = defaultdict(list)
        gm: torch.fx.GraphModule = torch.fx.symbolic_trace(module)

        # 遍历图中的每个节点，为每个节点添加标签，并构建标签到call_module节点的字典记录
        for node in gm.graph.nodes:
            # 根据节点名称的关键字设置标签
            if "linear1" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "linear2" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                # 如果节点是call_module类型，则记录到对应标签的列表中
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        
        # 返回更新后的图和标签到节点列表的字典
        return gm, tag_node
    # 定义一个名为 test_split_by_tags 的测试方法，没有参数，无返回值
    def test_split_by_tags(self) -> None:
        # 定义一个包含字符串 "red", "blue", "green" 的列表，表示标签
        tags = ["red", "blue", "green"]
        # 创建一个 TestSplitByTags.TestModule 的实例，命名为 module
        module = TestSplitByTags.TestModule()
        # 调用 TestSplitByTags.trace_and_tag 方法，将 module 和 tags 作为参数传入，
        # 返回结果保存在 gm 和 tag_node 变量中
        gm, tag_node = TestSplitByTags.trace_and_tag(module, tags)
        # 调用 split_by_tags 方法，传入 gm 和 tags 作为参数，同时设置 return_fqn_mapping=True，
        # 返回结果保存在 split_gm 和 orig_to_split_fqn_mapping 变量中
        split_gm, orig_to_split_fqn_mapping = split_by_tags(
            gm, tags, return_fqn_mapping=True
        )
        
        # 确保 split_gm 拥有且仅有按顺序命名的子模块 red_0, blue_1, green_2
        for idx, (name, _) in enumerate(split_gm.named_children()):
            if idx < len(tags):
                # 断言子模块的名称为 tags[idx]
                self.assertTrue(
                    name == tags[idx],
                    f"split_gm has an incorrect submodule named {name}",
                )

        # 确保每个子模块具有预期的（有序的）call_module 节点
        # 例如，名为 split_gm.red_0 的子模块只有 linear1；
        # 名为 split_gm.green_2 的子模块有 linear3 和 linear4，并且顺序正确
        sub_graph_idx = 0
        for sub_name, sub_graph_module in split_gm.named_children():
            node_idx = 0
            for node in sub_graph_module.graph.nodes:
                if node.op != "call_module":
                    continue
                # 断言节点的名称为 tag_node[f"{sub_name}"][node_idx]
                self.assertTrue(
                    node.name == tag_node[f"{sub_name}"][node_idx],
                    f"{sub_name} has incorrectly include {node.name}",
                )
                node_idx += 1
            sub_graph_idx += 1

        # 断言 orig_to_split_fqn_mapping 是否符合预期的映射关系
        self.assertEqual(
            orig_to_split_fqn_mapping,
            {
                "linear1": "red.linear1",
                "linear2": "blue.linear2",
                "linear3": "green.linear3",
                "linear4": "green.linear4",
            },
            f"{orig_to_split_fqn_mapping=}",
        )
class TestSplitOutputType(TestCase):
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 定义一个卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3，步长为1，包含偏置
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            # 定义ReLU激活函数层
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # 前向传播函数，先进行卷积操作
            conv = self.conv(x)
            # 将卷积结果乘以0.5
            conv = conv * 0.5
            # 对乘法结果进行ReLU激活
            relu = self.relu(conv)
            return relu

    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, inputs: torch.Tensor, tags: List[str]
    ) -> Tuple[torch.fx.GraphModule, Dict[str, List[str]]]:
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            conv - tag: "red"
            mul - tag: "blue"
            relu - tag: "green"

        At the beginning we have:
            gm:
                conv
                mul
                relu

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    conv
                blue:
                    mul
                green:
                    relu
        """
        # 创建一个默认字典，用于记录不同标签下的节点名称列表
        tag_node = defaultdict(list)
        # 使用torch.export.export导出模块，得到图模块gm
        gm: torch.fx.GraphModule = torch.export.export(module, (inputs,)).module()
        # 遍历图中的节点，根据节点名称添加相应的标签，并记录到tag_node中
        for node in gm.graph.nodes:
            if "conv" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "mul" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        # 返回标记后的图模块gm和标签到节点名称列表的字典tag_node
        return gm, tag_node

    def test_split_by_tags(self) -> None:
        # 定义标签列表
        tags = ["red", "blue", "green"]
        # 创建测试模块
        module = TestSplitOutputType.TestModule()
        # 生成随机输入数据
        inputs = torch.randn((1, 3, 224, 224))

        # 对模块进行追踪和标记，得到标记后的图模块gm和标签到节点名称列表的字典
        gm, tag_node = TestSplitOutputType.trace_and_tag(module, inputs, tags)
        # 根据标签分割图模块gm，同时返回原始节点名称到分割后节点全名的映射关系
        split_gm, orig_to_split_fqn_mapping = split_by_tags(
            gm, tags, return_fqn_mapping=True
        )

        # 分别对原始模块和分割后模块进行前向传播
        gm_output = module(inputs)
        split_gm_output = split_gm(inputs)

        # 断言：原始模块输出的类型与分割后模块输出的类型相同
        self.assertTrue(type(gm_output) == type(split_gm_output))
        # 断言：原始模块输出与分割后模块输出相等
        self.assertTrue(torch.equal(gm_output, split_gm_output))
```