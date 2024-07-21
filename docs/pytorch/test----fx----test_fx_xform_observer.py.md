# `.\pytorch\test\fx\test_fx_xform_observer.py`

```
# Owner(s): ["module: fx"]

# 导入必要的库和模块
import os
import tempfile

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

from torch.testing._internal.common_utils import TestCase

# 如果直接运行该文件，则抛出运行时错误提示信息
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestGraphTransformObserver，继承自 TestCase
class TestGraphTransformObserver(TestCase):
    
    # 定义测试方法 test_graph_transform_observer
    def test_graph_transform_observer(self):
        
        # 定义一个简单的神经网络模型 M，包含一个 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                # 对输入 x 进行 torch.neg 操作
                val = torch.neg(x)
                # 返回 torch.add 操作后的结果
                return torch.add(val, val)

        # 定义一个模式匹配函数 pattern，用于找出模型中的 torch.neg 操作
        def pattern(x):
            return torch.neg(x)

        # 定义一个替换函数 replacement，将 torch.neg 操作替换为 torch.relu 操作
        def replacement(x):
            return torch.relu(x)

        # 对模型 M 进行符号跟踪，得到 traced 模型
        traced = symbolic_trace(M())

        # 创建一个临时目录作为日志存储位置
        log_url = tempfile.mkdtemp()

        # 使用 GraphTransformObserver 监视 traced 模型的变换过程
        with GraphTransformObserver(traced, "replace_neg_with_relu", log_url) as ob:
            # 替换 traced 模型中的 torch.neg 操作为 torch.relu 操作
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            # 断言在观察器 ob 中创建了 "relu" 节点
            self.assertTrue("relu" in ob.created_nodes)
            # 断言观察器 ob 中消除了 "neg" 节点
            self.assertTrue("neg" in ob.erased_nodes)

        # 获取当前变换过程的计数
        current_pass_count = GraphTransformObserver.get_current_pass_count()

        # 断言在 log_url 目录下生成了输入图和输出图的 dot 文件
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_input_graph.dot",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_output_graph.dot",
                )
            )
        )
```