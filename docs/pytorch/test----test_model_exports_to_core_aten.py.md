# `.\pytorch\test\test_model_exports_to_core_aten.py`

```py
# 导入必要的模块和库
# Owner(s): ["oncall: mobile"]
import copy  # 导入copy模块，用于对象的深拷贝操作

import pytest  # 导入pytest模块，用于测试框架

import torch  # 导入PyTorch深度学习框架
import torch._export as export  # 导入PyTorch的导出模块

from torch.testing._internal.common_quantization import skip_if_no_torchvision  # 从内部测试工具中导入跳过无torchvision的装饰器
from torch.testing._internal.common_utils import TestCase  # 从通用测试工具中导入测试用例类


def _get_ops_list(m: torch.fx.GraphModule):
    # 定义一个函数用于获取图模块中的操作列表
    op_list = []
    # 遍历图中的每个节点
    for n in m.graph.nodes:
        # 如果节点的操作类型为"call_function"
        if n.op == "call_function":
            # 将节点的目标函数添加到操作列表中
            op_list.append(n.target)
    return op_list


class TestQuantizePT2EModels(TestCase):
    @pytest.mark.xfail()  # 使用pytest的xfail标记，表示预期测试失败
    @skip_if_no_torchvision  # 使用自定义装饰器，如果没有torchvision则跳过测试
    def test_vit_aten_export(self):
        from torchvision.models import vit_b_16  # 导入torchvision中的ViT模型

        m = vit_b_16(weights="IMAGENET1K_V1")  # 加载预训练的ViT模型
        m = m.eval()  # 将模型设置为评估模式
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)  # 生成一个示例输入
        m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))  # 捕获前自动求导图
        m(*example_inputs)  # 通过模型传递示例输入
        m = export.export(m, copy.deepcopy(example_inputs))  # 导出模型
        ops = _get_ops_list(m.graph_module)  # 获取导出后图模块中的操作列表
        non_core_aten_op_found = False  # 初始化非核心aten操作发现标志为False
        for op in ops:
            # 如果操作字符串中包含"scaled_dot_product"
            if "scaled_dot_product" in str(op):
                non_core_aten_op_found = True  # 设置非核心aten操作发现标志为True
        self.assertFalse(non_core_aten_op_found)  # 断言非核心aten操作未被发现


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests  # 从通用测试工具中导入运行测试的函数

    run_tests()  # 运行所有测试
```