# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_mixed_precision.py`

```py
# Owner(s): ["oncall: distributed"]

import sys  # 导入sys模块，用于访问系统相关功能
from typing import Dict, NamedTuple, Optional  # 导入类型提示相关模块

import torch  # 导入PyTorch深度学习框架
import torch.distributed as dist  # 导入PyTorch分布式通信模块
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable import fully_shard  # 导入PyTorch分布式计算组件
from torch.distributed.fsdp import MixedPrecision  # 导入PyTorch混合精度训练模块
from torch.testing._internal.common_distributed import (  # 导入PyTorch分布式测试相关模块
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import FSDPTest  # 导入PyTorch FSDP测试模块
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入PyTorch通用测试相关工具


if not dist.is_available():  # 如果当前环境不支持分布式功能
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印信息到标准错误流，表示跳过测试
    sys.exit(0)  # 终止程序执行

if TEST_WITH_DEV_DBG_ASAN:  # 如果在开发模式下使用了ASAN（地址检测）
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 打印相关问题说明到标准错误流
    )
    sys.exit(0)  # 终止程序执行


class TestMixedPrecision(FSDPTest):  # 定义一个测试类，继承自FSDPTest基类
    """Tests ``fully_shard`` with mixed precision."""  # 类的文档字符串，描述测试混合精度下的``fully_shard``功能

    class SubtestKey(NamedTuple):  # 定义一个命名元组子类SubtestKey
        cast_root_forward_inputs: bool  # 布尔值，表示是否进行根节点前向输入类型转换
        cast_forward_inputs_submodule: bool  # 布尔值，表示是否进行子模块前向输入类型转换
        use_root_no_params: bool  # 布尔值，表示是否使用根节点无参数

    class SubtestResult(NamedTuple):  # 定义一个命名元组子类SubtestResult
        subtest_alias: str  # 子测试别名，字符串类型
        model_dtype: torch.dtype = torch.float16  # 模型数据类型，默认为torch.float16
        c1_dtype: torch.dtype = torch.float16  # c1数据类型，默认为torch.float16
        c2_dtype: Optional[torch.dtype] = torch.float16  # c2数据类型，可选的torch数据类型，默认为torch.float16

    EXPECTED_CAST_DTYPES = {  # 预期的类型转换结果字典，映射SubtestKey到SubtestResult
        SubtestKey(True, True, True): SubtestResult(
            "cast_root_cast_child_no_root_params"  # 子测试别名为"cast_root_cast_child_no_root_params"
        ),
        SubtestKey(True, True, False): SubtestResult(
            "cast_root_cast_child_root_params",  # 子测试别名为"cast_root_cast_child_root_params"
            model_dtype=torch.float32,  # 模型数据类型为torch.float32
            c1_dtype=torch.float32,  # c1数据类型为torch.float32
        ),
        SubtestKey(True, False, True): SubtestResult(
            "cast_root_no_cast_child_no_root_params"  # 子测试别名为"cast_root_no_cast_child_no_root_params"
        ),
        SubtestKey(True, False, False): SubtestResult(
            "cast_root_no_cast_child_root_params",  # 子测试别名为"cast_root_no_cast_child_root_params"
            model_dtype=torch.float32,  # 模型数据类型为torch.float32
            c1_dtype=torch.float32,  # c1数据类型为torch.float32
        ),
        SubtestKey(False, True, True): SubtestResult(
            "no_cast_root_cast_child_no_root_params",  # 子测试别名为"no_cast_root_cast_child_no_root_params"
            model_dtype=torch.float32  # 模型数据类型为torch.float32
        ),
        SubtestKey(False, True, False): SubtestResult(
            "no_cast_root_cast_child_root_params",  # 子测试别名为"no_cast_root_cast_child_root_params"
            model_dtype=torch.float32,  # 模型数据类型为torch.float32
            c1_dtype=torch.float32,  # c1数据类型为torch.float32
        ),
        SubtestKey(False, False, True): SubtestResult(
            "no_cast_root_no_cast_child_no_root_params",  # 子测试别名为"no_cast_root_no_cast_child_no_root_params"
            model_dtype=torch.float32,  # 模型数据类型为torch.float32
            c1_dtype=torch.float32,  # c1数据类型为torch.float32
            c2_dtype=None,  # c2数据类型为None
        ),
        # SubtestKey(False, False, True): SubtestResult(  # 注释掉的代码段，待修复后期评估模式
        #     "no_cast_root_no_cast_child_no_root_params",
        #     model_dtype=torch.float32,
        #     c1_dtype=torch.float32,
        #     c2_dtype=torch.float32), # 修复评估模式后的预期结果
        SubtestKey(False, False, False): SubtestResult(
            "no_cast_root_no_cast_child_root_params",  # 子测试别名为"no_cast_root_no_cast_child_root_params"
            model_dtype=torch.float32,  # 模型数据类型为torch.float32
            c1_dtype=torch.float32,  # c1数据类型为torch.float32
            c2_dtype=torch.float32,  # c2数据类型为torch.float32
        ),
    }

    @property
    # 返回固定的全球尺寸为2
    def world_size(self):
        return 2

    # 如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_float16_cast_forward(self):
        # 运行子测试，测试浮点16位类型转换的前向传播
        self.run_subtests(
            {
                "cast_root_forward_inputs_submodule": [True, False],
                "cast_forward_inputs_submodule": [True, False],
                "use_root_no_params": [True, False],
            },
            self._test_float16_cast_forward,
        )

    # 执行浮点16位类型转换的前向传播测试
    def _test_float16_cast_forward(
        self,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        # 构建浮点转换的配置
        cast_forward_cfg = (
            cast_root_forward_inputs_submodule,
            cast_forward_inputs_submodule,
        )
        # 初始化输入和模型
        x, fsdp = self._input_and_model_init(*cast_forward_cfg, use_root_no_params)
        # 调用私有方法验证评估结果
        # self._validate_eval(x, fsdp)
        # 调用私有方法执行反向传播或验证错误
        self._backward_or_validate_error(x, fsdp, *cast_forward_cfg)
        # 调用私有方法断言期望的数据类型
        self._assert_expected_dtypes(fsdp, *cast_forward_cfg, use_root_no_params)

    # 初始化输入和模型
    def _input_and_model_init(
        self,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        # 创建空字典存储前向输入
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        # 创建混合精度对象，指定参数数据类型为float16
        float16 = MixedPrecision(
            param_dtype=torch.float16,
            cast_root_forward_inputs=cast_root_forward_inputs_submodule,
            cast_forward_inputs=cast_forward_inputs_submodule,
        )

        # 创建带有保存前向输入的模型，并移至CUDA设备
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        # 创建CUDA设备上的全零张量x
        x = torch.zeros(2, 100, device="cuda")

        # 将模型的一个子模块使用混合精度float16
        model.c2 = fully_shard(model.c2, mixed_precision=float16)

        # 如果不使用根节点没有参数，则一个float32模块将留给根节点
        if use_root_no_params:
            # 使用float16并包装所有子模块，使根节点没有直接参数
            model.c1 = fully_shard(model.c1, mixed_precision=float16)
            fsdp = fully_shard(model, mixed_precision=float16)
        else:
            fsdp = fully_shard(model)
        return x, fsdp

    # 验证评估模式总是强制全精度
    def _validate_eval(
        self, input: Dict[nn.Module, torch.Tensor], fsdp_model: nn.Module
    ):
        fsdp_model.eval()
        _ = fsdp_model(input)
        # 断言模型的前向输入数据类型为torch.float32
        self.assertEqual(fsdp_model.forward_inputs[fsdp_model.c1].dtype, torch.float32)
        self.assertEqual(fsdp_model.forward_inputs[fsdp_model.c2].dtype, torch.float32)
        fsdp_model.train()

    # 执行反向传播或验证错误
    def _backward_or_validate_error(
        self,
        input: Dict[nn.Module, torch.Tensor],
        fsdp_model: nn.Module,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
    ):
        # 检查是否需要在根模块或子模块中进行类型转换
        if not cast_root_forward_inputs_submodule and not cast_forward_inputs_submodule:
            # 如果两者都不需要进行类型转换，则断言引发 RuntimeError，并检查错误消息
            with self.assertRaisesRegex(
                RuntimeError,
                "mat1 and mat2 must have the same dtype",
            ):
                # 对输入进行求和并进行反向传播
                fsdp_model(input).sum().backward()
        else:
            # 如果需要进行类型转换，则只对输入进行求和并进行反向传播
            fsdp_model(input).sum().backward()

    def _assert_expected_dtypes(
        self,
        fsdp_model: nn.Module,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        # 创建子测试的键，用于确定预期的类型转换设置
        subtest_key = TestMixedPrecision.SubtestKey(
            cast_root_forward_inputs_submodule,
            cast_forward_inputs_submodule,
            use_root_no_params,
        )
        # 构建子测试失败时的错误消息
        subtest_fail_msg = f"Subtest `{TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key].subtest_alias}` failed."
        # 断言模型输入的 dtype 与预期的模型 dtype 相符
        self.assertEqual(
            fsdp_model.forward_inputs[fsdp_model].dtype,
            TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key].model_dtype,
            msg=subtest_fail_msg,
        )
        # 遍历模型的两个子模块 c1 和 c2
        for i, mod in enumerate((fsdp_model.c1, fsdp_model.c2), start=2):
            # 检查如果模型输入中包含该子模块，则断言其 dtype 与预期的 dtype 相符
            if fsdp_model.forward_inputs.get(mod, None) is not None:
                self.assertEqual(
                    fsdp_model.forward_inputs[mod].dtype,
                    TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key][i],
                    msg=subtest_fail_msg,
                )
# 如果当前脚本被直接执行（而不是被导入作为模块），则执行下面的代码
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```