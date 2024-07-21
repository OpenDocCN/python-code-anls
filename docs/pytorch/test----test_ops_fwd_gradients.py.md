# `.\pytorch\test\test_ops_fwd_gradients.py`

```py
# Owner(s): ["module: unknown"]

# 导入平台信息模块
import platform
# 导入偏函数模块
from functools import partial
# 导入 skipIf 别名为 skipif 的单元测试工具函数
from unittest import skipIf as skipif

# 导入 PyTorch 库
import torch
# 导入 PyTorch 内部测试相关模块
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
# 导入 PyTorch 内部测试共用方法数据库
from torch.testing._internal.common_methods_invocations import op_db

# 导入 PyTorch 内部共用测试工具函数
from torch.testing._internal.common_utils import (
    IS_MACOS,
    run_tests,
    skipIfTorchInductor,
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
)

# 在 macOS 上解决不稳定问题，限制线程数为 1
if IS_MACOS:
    torch.set_num_threads(1)

# 使用 ops 函数生成 _gradcheck_ops 函数，用于支持梯度检查的操作，使用双精度和复数双精度类型
_gradcheck_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble]
)

# 定义一个测试类 TestFwdGradients，继承自 TestGradients 类，并取消 Dynamo 严格测试的标记
@unMarkDynamoStrictTest
class TestFwdGradients(TestGradients):
    # 测试前向梯度和反向梯度是否正确计算
    @_gradcheck_ops(op_db)
    def test_fn_fwgrad_bwgrad(self, device, dtype, op):
        # 调用辅助方法 _skip_helper，跳过不支持的操作
        self._skip_helper(op, device, dtype)

        # 如果操作支持前向梯度和反向梯度
        if op.supports_fwgrad_bwgrad:
            # 调用辅助方法 _check_helper，进行前向-反向梯度计算的检查
            self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
        else:
            # 抛出未实现错误，如果尝试对不支持前向自动求导的操作进行计算
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = (
                "Running forward-over-backward gradgrad for an OP that has does not support it did not "
                "raise any error. If your op supports forward AD, you should set supports_fwgrad_bwgrad=True."
            )
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
    # 定义一个帮助函数，用于处理前向梯度检查
    def _forward_grad_helper(self, device, dtype, op, variant, is_inplace):
        # TODO: clean up how attributes are passed to gradcheck from OpInfos
        # 定义内部函数，调用梯度测试辅助函数，根据操作信息和是否原地操作确定是否检查批处理前向梯度
        def call_grad_test_helper():
            # 检查是否需要检查批处理前向梯度，根据操作的属性进行判断
            check_batched_forward_grad = (
                op.check_batched_forward_grad and not is_inplace
            ) or (op.check_inplace_batched_forward_grad and is_inplace)
            # 调用梯度测试辅助函数，传入相应参数进行测试
            self._grad_test_helper(
                device,
                dtype,
                op,
                variant,
                check_forward_ad=True,
                check_backward_ad=False,
                check_batched_grad=False,
                check_batched_forward_grad=check_batched_forward_grad,
            )

        # 如果操作支持前向自动求导，则直接调用梯度测试辅助函数
        if op.supports_forward_ad:
            call_grad_test_helper()
        else:
            # 否则，预期会引发 NotImplementedError 异常，捕获并检查异常消息
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = (
                "Running forward AD for an OP that has does not support it did not "
                "raise any error. If your op supports forward AD, you should set supports_forward_ad=True"
            )
            # 使用断言检查是否抛出了预期的异常
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                call_grad_test_helper()

    # 使用装饰器声明使用指定的梯度检查操作集合，并跳过在s390x架构上的特定测试
    @_gradcheck_ops(op_db)
    @skipif(
        platform.machine() == "s390x",
        reason="Different precision of openblas functions: https://github.com/OpenMathLib/OpenBLAS/issues/4194",
    )
    # 前向模式自动求导测试函数，用于检查前向自动求导是否正常工作
    def test_forward_mode_AD(self, device, dtype, op):
        # 跳过测试，如果操作被标记为需要跳过
        self._skip_helper(op, device, dtype)

        # 调用前向梯度辅助函数，测试非原地操作的前向自动求导
        self._forward_grad_helper(device, dtype, op, op.get_op(), is_inplace=False)

    # 使用装饰器声明使用指定的梯度检查操作集合，并跳过 Torch Inductor 上尚待修复的测试
    @_gradcheck_ops(op_db)
    @skipIfTorchInductor("to be fixed")
    # 原地前向模式自动求导测试函数，用于检查支持原地操作的前向自动求导是否正常工作
    def test_inplace_forward_mode_AD(self, device, dtype, op):
        # 跳过测试，如果操作未声明支持原地操作或者不支持原地自动求导
        self._skip_helper(op, device, dtype)

        # 如果操作不支持原地变体或者原地自动求导，则跳过此测试
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")

        # 调用前向梯度辅助函数，测试原地操作的前向自动求导
        self._forward_grad_helper(
            device, dtype, op, self._get_safe_inplace(op.get_inplace()), is_inplace=True
        )
# 调用函数 instantiate_device_type_tests，用于实例化设备类型测试，将 TestFwdGradients 绑定到全局环境中
instantiate_device_type_tests(TestFwdGradients, globals())

# 检查当前脚本是否作为主程序执行
if __name__ == "__main__":
    # 启用 TestCase 的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试函数
    run_tests()
```