# `.\pytorch\test\fx\test_shape_inference.py`

```
# 导入必要的模块和库，包括unittest、torch等
# Owner(s): ["module: fx"]
import copy  # 导入copy模块，用于复制对象
import unittest  # 导入unittest模块，用于编写和运行测试
from collections import defaultdict  # 导入defaultdict类，用于创建默认字典

import torch  # 导入PyTorch库
import torch.fx as fx  # 导入torch.fx模块，用于构建和分析函数图
from torch._dynamo.source import LocalSource  # 导入LocalSource类，可能用于源码跟踪
from torch.fx.experimental.shape_inference.infer_shape import infer_shape  # 导入infer_shape函数，用于形状推断
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,  # 导入infer_symbol_values函数，用于符号值推断
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv  # 导入DimDynamic和ShapeEnv类，用于符号形状推断

# 定义测试类TestShapeInference，继承自unittest.TestCase，用于测试形状推断功能
class TestShapeInference(unittest.TestCase):
    def test_infer_symbol_values(self):
        # 定义内部函数 mksym，用于创建符号和符号整数节点
        def mksym(shape_env, value, source, dynamic_dim) -> None:
            return shape_env.create_symintnode(
                shape_env.create_symbol(
                    value,
                    source=source,
                    dynamic_dim=dynamic_dim,
                ),
                hint=value,
                source=source,
            )
        
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 设置符号数量 N
        N = 8
        # 创建示例字典 sample，包含键 s0 到 s7，每个键对应值为 2
        sample = {f"s{i}": 2 for i in range(N)}
        # 初始化符号整数节点列表 init_symints
        init_symints = [
            # 调用 mksym 函数创建符号整数节点，dynamic_dim 为 DimDynamic.DYNAMIC
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        # 深拷贝初始符号整数节点列表，得到 symints
        symints = copy.deepcopy(init_symints)
        # 创建符号到索引的映射字典 symbol_to_idx_dict
        symbol_to_idx_dict = {f"s{i}": i for i in range(N)}
        # 创建默认值为空列表的填充约束字典 padding_constraints
        padding_constraints = defaultdict(list)

        # 准备约束字符串列表 constraints
        constraints = []
        constraints.append(
            "The size of tensor a (s1) must match the size of tensor b (1773) at non-singleton dimension 1)"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, (s2//2) + 12] but got: [s0, 120]."
        )
        constraints.append("shape '[s0, -1, 32]' is invalid for input of size s0*s3")
        constraints.append(
            "a and b must have same reduction dim, but got [32*s0, s3] X [20, 15]."
        )
        constraints.append(
            "a and b must have same reduction dim, but got [s0, s4 + 1568] X [5728, 1024]."
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 40] but got: [s0, s5]."
        )
        constraints.append(
            "shape '[s0, -1, 32]' is invalid for input of size s0*s6 + 1344*s0"
        )
        constraints.append(
            "shape '[-1, 47]' is invalid for input of size 32*s0*s6 + 1344*s0"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 47*s6] but got: [s0*s6, 47]."
        )
        constraints.append("Split sizes add up to 4258 but got the tensor's size of s7")

        # 对每个约束字符串调用 infer_symbol_values 函数进行推断
        for constraint in constraints:
            infer_symbol_values(
                symints,
                init_symints,
                symbol_to_idx_dict,
                padding_constraints,
                constraint,
            )

        # 断言检查推断后的符号整数节点值是否符合预期
        self.assertEqual(symints[1], 1773)
        self.assertEqual(symints[2], 216)
        self.assertEqual(symints[3], 640)
        self.assertEqual(symints[4], 4160)
        self.assertEqual(symints[5], 40)
        self.assertEqual(symints[6], 160)
        self.assertEqual(symints[7], 4258)
    def test_infer_shape(self):
        # 定义一个测试用的神经网络模块
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化第一层权重和偏置
                self.w_1 = torch.empty([256, 328])
                self.b_1 = torch.empty([256])
                # 初始化第二层权重和偏置
                self.w_2 = torch.empty([328, 256])
                self.b_2 = torch.empty([328])

            def forward(self, x):
                # 计算第一层的线性变换
                l_1 = torch.nn.functional.linear(x, self.w_1, bias=self.b_1)
                # 对第一层输出应用 sigmoid 激活函数
                s_1 = torch.sigmoid(l_1)
                # 计算第二层的线性变换
                l_2 = torch.nn.functional.linear(s_1, self.w_2, bias=self.b_2)
                # 对第二层输出应用 tanh 激活函数
                t_1 = torch.tanh(l_2)
                # 返回最终输出
                return t_1

        def generate_graph_module(model):
            # 对输入的模型进行符号化追踪
            gm = fx.symbolic_trace(model)
            return gm

        # 创建一个 TestModule 实例
        m = TestModule()
        # 对 TestModule 进行符号化追踪，生成符号图模块
        gm = generate_graph_module(m)
        # 创建一个包含随机输入张量的列表
        input_tensors = [torch.randn(1, 1)]
        # 对符号图模块进行推断形状
        infer_shape(gm, input_tensors)
```