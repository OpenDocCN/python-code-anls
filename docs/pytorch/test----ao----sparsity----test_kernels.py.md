# `.\pytorch\test\ao\sparsity\test_kernels.py`

```py
# Owner(s): ["module: unknown"]

import copy  # 导入深拷贝模块，用于复制对象
import io  # 导入输入输出模块，用于处理字节流
import logging  # 导入日志记录模块，用于输出日志信息
from itertools import product  # 导入 product 函数，用于生成迭代器的笛卡尔积

import numpy as np  # 导入 NumPy 库，用于数值计算

import torch  # 导入 PyTorch 深度学习库
import torch.ao.quantization as tq  # 导入 PyTorch AO（量化优化）模块

from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.ao.pruning.sparsifier.utils import fqn_to_module  # 从稀疏化剪枝工具中导入函数
from torch.testing._internal.common_quantized import (  # 导入通用量化测试相关模块和函数
    override_cpu_allocator_for_qnnpack,
    override_qengines,
    qengine_is_fbgemm,
    qengine_is_onednn,
    qengine_is_qnnpack,
    qengine_is_x86,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase  # 导入测试工具函数和类

# TODO: Once more test files are created, move the contents to a ao folder.

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# 定义测试类 TestQuantizedSparseKernels，继承自 TestCase 类
class TestQuantizedSparseKernels(TestCase):
    
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    @override_qengines
    def _sparse_layer_test_helper(
        model_class,
        sparse_mapping,
        ref_mapping,
        qconfig_dict,
        fqn_to_check,
        test_class,
        test_scripting,
    ):
        # SET UP TEST PARAMETERS, INPUTS AND WEIGHTS
        # ------------------------------------------
        batch_size = 12  # 定义批量大小为 12
        input_channels = 4  # 定义输入通道数为 4
        output_channels = 7  # 定义输出通道数为 7
        model = model_class(input_channels, output_channels)  # 使用给定的 model_class 创建模型对象

        # For sparse kernels both the activation and weight ZP = 0
        X_scale = 0.2  # 定义输入的缩放因子为 0.2
        X_zp = 2  # 定义输入的零点值为 2
        W_scale = 1e-2  # 定义权重的缩放因子为 0.01
        W_zp = 0  # 定义权重的零点值为 0

        X_fp32 = torch.randn(batch_size, input_channels, dtype=torch.float32)  # 生成随机的输入张量 X_fp32
        float_bias = torch.randn(output_channels, dtype=torch.float32)  # 生成随机的浮点偏置项 float_bias

        # generate a weight which we'll insert into the model
        W_fp32 = torch.randn(output_channels, input_channels, dtype=torch.float32)  # 生成随机的权重张量 W_fp32
        mask = torch.randint(0, 2, W_fp32.shape)  # 生成随机的 0/1 掩码张量
        W_fp32 *= mask  # 将权重张量 W_fp32 与掩码张量相乘

# 定义稀疏量化模型类 SparseQuantizedModel，继承自 nn.Module 类
class SparseQuantizedModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)  # 初始化线性层

    def forward(self, x):
        return self.linear(x)  # 前向传播函数，应用线性层

# 定义测试类 TestQuantizedSparseLayers，继承自 TestCase 类
class TestQuantizedSparseLayers(TestCase):
    
    @override_qengines
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_sparse_qlinear_serdes(self):
        # 注意：目前对于稀疏内核，fbgemm仅支持静态量化的稀疏线性层，
        # qnnpack仅支持动态量化的稀疏线性层。
        # 因此我们有两种不同的测试。
        # fbgemm测试静态流程，qnnpack测试动态流程。
        # 后续应统一这些，并适当修复测试。
        
        # 使用 SparseQuantizedModel 类作为被测试的模型类
        model_class = SparseQuantizedModel
        # 要检查的量化线性层的全限定名称
        fqn_to_check = "linear"
        
        # 根据当前量化引擎选择稀疏映射和参考映射
        if qengine_is_fbgemm():
            # 获取默认的静态稀疏量化模块映射
            sparse_mapping = tq.get_default_static_sparse_quant_module_mappings()
            # 获取默认的静态量化模块映射
            ref_mapping = tq.get_default_static_quant_module_mappings()
            # 设置线性层的默认fbgemm量化配置
            qconfig_dict = {nn.Linear: tq.get_default_qconfig("fbgemm")}
        elif qengine_is_qnnpack():
            # 获取默认的动态稀疏量化模块映射
            sparse_mapping = tq.get_default_dynamic_sparse_quant_module_mappings()
            # 获取默认的动态量化模块映射
            ref_mapping = tq.get_default_dynamic_quant_module_mappings()
            # 设置线性层的默认qnnpack动态量化配置
            qconfig_dict = {nn.Linear: tq.qconfig.default_dynamic_qconfig}
        else:
            # 如果没有匹配的量化引擎，直接返回
            return

        # 调用辅助函数进行稀疏层测试
        _sparse_layer_test_helper(
            model_class=model_class,
            sparse_mapping=sparse_mapping,
            ref_mapping=ref_mapping,
            qconfig_dict=qconfig_dict,
            fqn_to_check=fqn_to_check,
            test_class=self,
            test_scripting=True,
        )
# 如果当前模块是主程序（即没有被导入到其它模块），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码或功能
    run_tests()
```