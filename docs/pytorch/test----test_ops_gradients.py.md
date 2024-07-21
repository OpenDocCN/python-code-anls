# `.\pytorch\test\test_ops_gradients.py`

```py
# Owner(s): ["module: unknown"]

from functools import partial  # 导入 partial 函数，用于创建带有部分参数的新函数

import torch  # 导入 PyTorch 库
from torch.testing._internal.common_device_type import (  # 导入设备类型测试相关模块和函数
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db  # 导入操作方法的测试数据库

from torch.testing._internal.common_utils import (  # 导入通用测试工具函数和类
    run_tests,
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.custom_op_db import custom_op_db  # 导入自定义操作数据库
from torch.testing._internal.hop_db import hop_db  # 导入高阶操作数据库

# gradcheck 需要双精度
_gradcheck_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble]
)

@unMarkDynamoStrictTest  # 装饰器，用于取消 DynamoStrict 测试标记
class TestBwdGradients(TestGradients):  # 定义 TestBwdGradients 类，继承自 TestGradients 类

    @_gradcheck_ops(op_db + hop_db + custom_op_db)  # 使用 _gradcheck_ops 对 op_db、hop_db 和 custom_op_db 中的操作进行梯度检查
    def test_fn_grad(self, device, dtype, op):
        # Tests that gradients are computed correctly
        # 验证梯度是否正确计算，具体由 test_dtypes 在 test_ops.py 中进行验证
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Dtype is not in supported backward dtypes!")
        else:
            self._grad_test_helper(device, dtype, op, op.get_op())

    # Method grad (and gradgrad, see below) tests are disabled since they're
    #   costly and redundant with function grad (and gradgad) tests
    # 由于它们耗时且与函数 grad (和 gradgrad，见下文) 测试重复，因此禁用了方法 grad 测试
    # @_gradcheck_ops(op_db)
    # def test_method_grad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._grad_test_helper(device, dtype, op, op.get_method())

    @_gradcheck_ops(op_db + custom_op_db)  # 使用 _gradcheck_ops 对 op_db 和 custom_op_db 中的操作进行梯度检查
    def test_inplace_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)  # 辅助函数，用于跳过特定操作的测试
        if not op.inplace_variant:
            self.skipTest("Op has no inplace variant!")  # 如果操作没有原地变体，则跳过测试

        # Verifies an operation doesn't support inplace autograd if it claims not to
        # 如果操作声明不支持原地自动求导，则验证该操作不支持原地自动求导
        if not op.supports_inplace_autograd:
            inplace = self._get_safe_inplace(op.get_inplace())  # 获取安全的原地操作函数
            for sample in op.sample_inputs(device, dtype, requires_grad=True):
                if sample.broadcasts_input:
                    continue
                with self.assertRaises(Exception):  # 验证是否会引发异常
                    result = inplace(sample)
                    result.sum().backward()
        else:
            self._grad_test_helper(  # 执行梯度测试助手函数
                device, dtype, op, self._get_safe_inplace(op.get_inplace())
            )

    @_gradcheck_ops(op_db + hop_db + custom_op_db)  # 使用 _gradcheck_ops 对 op_db、hop_db 和 custom_op_db 中的操作进行梯度检查
    def test_fn_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)  # 辅助函数，用于跳过特定操作的测试
        if not op.supports_gradgrad:
            self.skipTest(
                "Op claims it doesn't support gradgrad. This is not verified."
            )
        else:
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")  # 执行检查助手函数，验证梯度的梯度是否正确计算

    # Test that gradients of gradients are properly raising
    @_gradcheck_ops(op_db + custom_op_db)  # 使用 _gradcheck_ops 对 op_db 和 custom_op_db 中的操作进行梯度检查
    # 定义一个测试方法，测试操作在给定设备和数据类型上的反向二阶导数
    def test_fn_fail_gradgrad(self, device, dtype, op):
        # 调用辅助方法，根据操作跳过不支持的设备和数据类型组合
        self._skip_helper(op, device, dtype)
        # 如果操作支持二阶梯度，则跳过测试并显示相应信息
        if op.supports_gradgrad:
            self.skipTest("Skipped! Operation does support gradgrad")

        # 设置错误消息的正则表达式，用于断言捕获运行时异常
        err_msg = r"derivative for .* is not implemented"
        # 使用断言确保运行时异常抛出，并匹配指定的错误消息
        with self.assertRaisesRegex(RuntimeError, err_msg):
            # 调用辅助方法，检查操作在给定设备和数据类型上的反向二阶导数
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")

    # gradgrad 方法（以及上面的 grad 方法）测试已禁用，因为它们成本高且与函数 gradgrad（和 grad）测试冗余
    # @_gradcheck_ops(op_db)
    # def test_method_gradgrad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._gradgrad_test_helper(device, dtype, op, op.get_method())

    # 使用装饰器 _gradcheck_ops(op_db) 标记的方法，用于测试支持就地操作二阶导数的操作
    @_gradcheck_ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        # 调用辅助方法，根据操作跳过不支持的设备和数据类型组合
        self._skip_helper(op, device, dtype)
        # 如果操作不支持原地变体或不支持原地自动微分，则跳过测试并显示相应信息
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        # 调用辅助方法，检查操作在给定设备和数据类型上的原地二阶导数
        self._check_helper(
            device, dtype, op, self._get_safe_inplace(op.get_inplace()), "bwgrad_bwgrad"
        )
# 调用函数 instantiate_device_type_tests，并传入 TestBwdGradients 和 globals() 参数进行实例化设备类型测试
instantiate_device_type_tests(TestBwdGradients, globals())

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 启用 TestCase 类的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试
    run_tests()
```