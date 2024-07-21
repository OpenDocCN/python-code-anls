# `.\pytorch\test\test_ops_jit.py`

```py
# 导入所需模块和函数
from functools import partial  # 导入 partial 函数，用于创建偏函数
from textwrap import dedent  # 导入 dedent 函数，用于去除多行字符串的缩进

import torch  # 导入 PyTorch 模块

# 导入测试相关模块和函数
from torch.testing import FileCheck  # 导入 FileCheck 用于文件检查
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,  # 导入 instantiate_device_type_tests 用于实例化设备类型测试
    OpDTypes,  # 导入 OpDTypes，定义操作的数据类型
    ops,  # 导入 ops，用于执行操作
)
from torch.testing._internal.common_jit import (
    check_against_reference,  # 导入 check_against_reference，用于检查引用
    JitCommonTestCase,  # 导入 JitCommonTestCase 作为基类
)
from torch.testing._internal.common_methods_invocations import op_db  # 导入 op_db，用于操作数据库
from torch.testing._internal.common_utils import (
    clone_input_helper,  # 导入 clone_input_helper，用于克隆输入数据辅助函数
    first_sample,  # 导入 first_sample，用于获取第一个样本
    IS_SANDCASTLE,  # 导入 IS_SANDCASTLE，用于检查是否在沙堡环境下运行
    run_tests,  # 导入 run_tests，用于运行测试
    TestCase,  # 导入 TestCase，定义测试用例
    unMarkDynamoStrictTest,  # 导入 unMarkDynamoStrictTest，用于取消 Dynamo 严格测试标记
)
from torch.testing._internal.jit_metaprogramming_utils import (
    check_alias_annotation,  # 导入 check_alias_annotation，用于检查别名注解
    create_script_fn,  # 导入 create_script_fn，用于创建脚本函数
    create_traced_fn,  # 导入 create_traced_fn，用于创建跟踪函数
)
from torch.testing._internal.jit_utils import (
    disable_autodiff_subgraph_inlining,  # 导入 disable_autodiff_subgraph_inlining，用于禁用自动微分子图内联
    is_lambda,  # 导入 is_lambda，用于检查是否为 lambda 函数
)

# 声明 _variant_ops 为 ops 函数的偏函数，限定操作的数据类型为 torch.float 和 torch.cfloat
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)


# 定义 TestJit 类，继承自 JitCommonTestCase，用于测试 JIT 和 eager 的操作一致性，
# 同时检查 JIT 特定的别名模式和预期的自动微分行为。
@unMarkDynamoStrictTest  # 取消 Dynamo 严格测试标记
class TestJit(JitCommonTestCase):
    exact_dtype = True  # 设置确切的数据类型匹配

    # 定义测试方法，验证操作的前向和后向传播在不同的操作变体（函数、方法、inplace）和运行时（eager、跟踪、脚本化）下产生相同的值。
    # 注意：当前未测试 inplace x {traced, scripted} 组合
    @_variant_ops(op_db)
    # 定义一个测试函数，用于验证操作的不同变体在 JIT 编译下的一致性
    def test_variant_consistency_jit(self, device, dtype, op):
        # 根据数据类型确定是否需要梯度
        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )

        # 确定是否需要包含共轭输入样本
        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        # 获取操作的样本输入
        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=_requires_grad,
            include_conjugated_inputs=include_conjugated_inputs,
        )

        # 获取操作的函数和方法作为测试的不同变体
        func = op.get_op()
        method = op.get_method()
        variants = {
            "function": func,
            "method": method,
        }

        # 如果操作的函数类型是 torch._ops.OpOverload 类型，则跳过测试
        if isinstance(func, torch._ops.OpOverload):
            self.skipTest("variant consistency doesn't work on torch.ops")

        # 如果操作名字在指定列表中，标记为具有伪造函数
        has_fake_function = op.name in ["resize_", "resize_as_"]

        # 如果具有伪造函数，则仅使用相应的方法作为变体，并生成不需要梯度的输入样本
        if has_fake_function:
            variants = {"method": getattr(torch.Tensor, op.name)}
            samples = op.sample_inputs(device, dtype, requires_grad=False)

        tested = False
        # 遍历样本输入进行测试
        for sample in samples:
            # 遍历测试不同的函数类型和对应的变体
            for func_type, variant in variants.items():
                if variant is None:
                    continue

                # 对于 lambda 函数，由于脚本化和别名分析不适用，跳过测试
                if is_lambda(variant):
                    continue

                tested = True
                try:
                    # 执行单个变体的 JIT 测试
                    self.indiv_variant_test_jit(
                        device, dtype, op, sample, func_type, variant, has_fake_function
                    )
                except Exception as e:
                    # 发生异常时，生成错误信息并抛出异常
                    variant_error_info = dedent(
                        f"""
                        Error testing {op.name} {func_type} variant
                        with dtype: {dtype}
                        with inputs {sample}:
                    """
                    )
                    raise Exception(variant_error_info) from e  # noqa: TRY002

        # 如果没有执行任何逻辑，则断言测试失败
        assert tested, "JIT Test does not execute any logic"

    # 定义单个变体 JIT 测试函数
    def indiv_variant_test_jit(
        self, device, dtype, op, sample, func_type, variant, has_fake_function
    ):
        # alias testing 仅针对 torch.float 类型进行测试
        _alias_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float,))

    # 使用部分应用的 ops 函数来测试别名操作
    @_alias_ops(op for op in op_db if op.aliases)
# 调用函数实例化设备类型测试，传入 TestJit 类和全局变量字典
instantiate_device_type_tests(TestJit, globals())

# 如果当前脚本被作为主程序执行
if __name__ == "__main__":
    # 启用 TestCase 类的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行测试
    run_tests()
```