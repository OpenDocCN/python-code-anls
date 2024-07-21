# `.\pytorch\test\distributed\fsdp\test_fsdp_apply.py`

```py
# Owner(s): ["oncall: distributed"]  # 代码所有者，指定分布式相关责任人员

import sys  # 导入系统模块，用于处理系统相关功能

import torch  # 导入PyTorch深度学习库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入FSDP分布式训练模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试用例辅助函数
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,  # 导入CUDA初始化模式枚举
    FSDPInitMode,  # 导入FSDP初始化模式枚举
    FSDPTest,  # 导入FSDP测试基类
    NestedWrappedModule,  # 导入嵌套封装模块类
    TransformerWithSharedParams,  # 导入具有共享参数的Transformer模块类
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行函数和测试开关

if not dist.is_available():  # 如果分布式环境不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印信息并输出到标准错误流
    sys.exit(0)  # 退出程序，返回状态码0

if TEST_WITH_DEV_DBG_ASAN:  # 如果开启了开发调试ASAN
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 打印相关信息到标准错误流
    )
    sys.exit(0)  # 退出程序，返回状态码0

class TestApply(FSDPTest):  # 定义测试类TestApply，继承自FSDPTest基类
    @property
    def world_size(self):  # 定义属性world_size，返回值为2
        return 2

    @torch.no_grad()  # 使用torch.no_grad()上下文管理器，禁止梯度计算
    def _init_linear_weights(self, m):  # 定义函数_init_linear_weights，初始化线性层的权重和偏置
        if type(m) == nn.Linear:  # 如果模块类型为nn.Linear
            m.weight.fill_(1.0)  # 设置权重为1.0
            m.bias.fill_(1.0)  # 设置偏置为1.0

    def check_weights(self, fsdp, expected_tensor_fn, check):  # 定义检查权重的函数
        with FSDP.summon_full_params(fsdp, recurse=True):  # 使用FSDP的上下文管理器，获取所有参数
            linear_modules = [
                module for module in fsdp.modules() if type(module) == nn.Linear
            ]  # 获取所有包含nn.Linear模块的列表
            for module in linear_modules:  # 遍历线性层模块
                for param in module.parameters():  # 遍历每个模块的参数
                    expected = expected_tensor_fn(param)  # 根据函数expected_tensor_fn计算期望值
                    check(param, expected, f"Got {param} but expected {expected}")  # 检查参数是否符合期望值

    def _check_apply(self, fsdp):  # 定义应用函数_apply，用于测试应用操作的效果
        # Assert linear weights are not all 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertNotEqual
        )  # 断言线性层的权重不全为1.0

        fsdp.apply(self._init_linear_weights)  # 对fsdp应用_init_linear_weights方法

        # Ensure all weights are 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertEqual
        )  # 确保所有权重都为1.0

    @skip_if_lt_x_gpu(2)  # 如果GPU数量少于2，则跳过测试
    def test_nested_module_apply(self):  # 测试嵌套模块的应用操作
        """Tests that ``apply()`` modifies parameter values in-place on a
        non-FSDP-root nested FSDP-wrapped model."""
        nested_wrapped_module = NestedWrappedModule.init(  # 初始化嵌套封装模块
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
        )
        self._check_apply(nested_wrapped_module)  # 调用_check_apply方法对嵌套模块进行应用测试

    @skip_if_lt_x_gpu(2)  # 如果GPU数量少于2，则跳过测试
    def test_transformer_module_apply(self):  # 测试Transformer模块的应用操作
        """Tests that ``apply()`` modifies parameter values in-place on an
        FSDP-wrapped transformer model with shared parameters."""
        transformer = TransformerWithSharedParams.init(  # 初始化具有共享参数的Transformer模块
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
        )
        self._check_apply(transformer)  # 调用_check_apply方法对Transformer模块进行应用测试

    @skip_if_lt_x_gpu(2)  # 如果GPU数量少于2，则跳过测试
    def test_apply_in_summon_raises_error(self):
        """测试在 ``summon_full_params()`` 上下文中调用 FSDP 实例的 ``apply()`` 是否会引发错误。"""
        # 初始化一个带有共享参数的 Transformer 实例
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_AFTER,
        )
        # 进入 ``summon_full_params()`` 上下文，并传入 transformer 自身作为参数
        with transformer.summon_full_params(transformer):
            # 断言在此上下文中调用 ``apply()`` 方法会引发 ValueError 异常，并包含特定错误信息
            with self.assertRaisesRegex(ValueError, "expected to be in states"):
                transformer.apply(self._init_linear_weights)
# 如果当前模块被直接运行（而不是被导入到其他模块中），执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码或测试套件
    run_tests()
```