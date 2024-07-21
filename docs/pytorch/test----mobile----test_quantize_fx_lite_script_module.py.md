# `.\pytorch\test\mobile\test_quantize_fx_lite_script_module.py`

```
# Owner(s): ["oncall: mobile"]

# 导入需要的库和模块
import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.utils.bundled_inputs
from torch.ao.quantization import default_qconfig, float_qparams_weight_only_qconfig

# 导入量化相关的模块和函数
# 基于 FX 图模式的量化
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    NodeSpec as ns,
    QuantizationLiteTestCase,
)

# 定义一个测试类，继承自 QuantizationLiteTestCase
class TestLiteFuseFx(QuantizationLiteTestCase):
    # 测试方法来自于 ./caffe2/test/quantization/fx/test_quantize_fx.py

    # 测试嵌入层量化
    def test_embedding(self):
        # 定义一个包含嵌入层的模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        # 创建并评估模型
        model = M().eval()
        # 生成随机整数索引
        indices = torch.randint(low=0, high=10, size=(20,))

        # 定义量化配置和节点
        quantized_node = ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        # 遍历配置，准备和转换模型，并比较 FX 脚本和移动端行为
        for qconfig, node in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randint(low=0, high=10, size=(20,)),
            )
            m = convert_fx(m)
            self._compare_script_and_mobile(m, input=indices)

    # 测试二维卷积量化
    def test_conv2d(self):
        # 定义一个包含两个卷积层的模型类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # 创建并评估模型
        m = M().eval()
        # 定义量化配置字典，其中第一个卷积层 quantized，第二个卷积层未 quantized
        qconfig_dict = {"": default_qconfig, "module_name": [("conv1", None)]}
        m = prepare_fx(m, qconfig_dict, example_inputs=torch.randn(1, 1, 1, 1))
        data = torch.randn(1, 1, 1, 1)
        m = convert_fx(m)
        # 比较 FX 脚本和移动端行为，验证第一个卷积层 quantized，第二个卷积层未 quantized
        self._compare_script_and_mobile(m, input=data)
    def test_submodule(self):
        # 定义测试子模块的函数
        # 测试量化完整模块、子模块和线性层
        configs = [
            {},  # 空配置
            {"module_name": [("subm", None)]},  # 配置只量化名为 "subm" 的模块
            {"module_name": [("fc", None)]},   # 配置只量化名为 "fc" 的模块
        ]
        # 遍历不同的配置
        for config in configs:
            # 创建并转换为评估模式的 LinearModelWithSubmodule 模型
            model = LinearModelWithSubmodule().eval()
            # 构建量化配置字典，使用 qnnpack 的默认量化配置
            qconfig_dict = {
                "": torch.ao.quantization.get_default_qconfig("qnnpack"),
                **config,  # 将当前配置合并到量化配置字典中
            }
            # 使用 prepare_fx 函数准备模型，提供示例输入为 5x5 的随机张量
            model = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randn(5, 5),
            )
            # 将准备好的模型转换为量化模型
            quant = convert_fx(model)

            # 创建输入为 5x5 的随机张量 x
            x = torch.randn(5, 5)
            # 调用 _compare_script_and_mobile 方法，比较量化模型 quant 的输出和输入 x
            self._compare_script_and_mobile(quant, input=x)
# 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码
if __name__ == "__main__":
    run_tests()  # 调用名为 run_tests 的函数进行测试，忽略 F821 错误
```