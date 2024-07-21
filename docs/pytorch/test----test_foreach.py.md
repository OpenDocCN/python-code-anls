# `.\pytorch\test\test_foreach.py`

```
# Owner(s): ["module: mta"]

# 引入需要的库和模块
import itertools  # 提供用于迭代工具的函数
import os  # 提供与操作系统交互的功能
import random  # 提供生成随机数的功能
import re  # 提供正则表达式操作的支持
import unittest  # 提供单元测试框架
import weakref  # 提供弱引用对象的支持
from contextlib import nullcontext  # 提供上下文管理工具，用于创建一个空的上下文
from numbers import Number  # 提供处理数字类型的基类

import torch  # 引入PyTorch库

from torch.testing import make_tensor  # 引入创建测试张量的函数
from torch.testing._comparison import default_tolerances  # 引入默认比较容差
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 引入多GPU测试相关的常量
from torch.testing._internal.common_device_type import (  # 引入设备类型相关的函数和常量
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_dtype import (  # 引入数据类型相关的函数和常量
    all_types_and_complex_and,
    floating_types,
    floating_types_and,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (  # 引入方法调用相关的函数
    foreach_binary_op_db,
    foreach_other_op_db,
    foreach_pointwise_op_db,
    foreach_reduce_op_db,
    foreach_unary_op_db,
)
from torch.testing._internal.common_utils import (  # 引入通用测试工具函数
    gradcheck,
    parametrize,
    run_tests,
    skipIfRocmVersionLessThan,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    TestCase,
)

# 错误消息常量，用于布尔类型操作的错误提示
_BOOL_SUB_ERR_MSG = "Subtraction, the `-` operator"


class RegularFuncWrapper:
    def __init__(self, func):
        self.func = func  # 初始化实例时存储传入的函数对象

    def __call__(self, inputs, scalars=None, **kwargs):
        if scalars is not None:
            assert len(inputs) == 3  # 检查输入的列表长度是否为3
            # 如果提供了标量参数，则将每个标量分配给正常函数，并且需要特别考虑它作为正常函数的关键字参数
            # （奇怪的是，在foreach函数中它不是关键字参数）
            return [
                self.func(*i, value=scalars[idx], **kwargs)
                for idx, i in enumerate(zip(*inputs))
            ]
        if len(inputs) == 2 and isinstance(inputs[1], (Number, torch.Tensor)):
            # 如果输入的列表长度为2，并且第二个输入是数字或者张量，则将其转换为与第一个输入相同长度的列表
            inputs[1] = [inputs[1] for _ in range(len(inputs[0]))]
        # 对输入进行迭代处理，并调用存储的函数进行处理
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class ForeachFuncWrapper:
    def __init__(self, func):
        self.func = func  # 初始化实例时存储传入的函数对象
        # 某些foreach函数没有原地实现
        self.is_inplace = False if func is None else func.__name__.endswith("_")
    # 定义一个调用函数，用于执行特定操作
    def __call__(self, inputs, is_cuda, expect_fastpath, **kwargs):
        # 初始化 actual 变量为 None
        actual = None
        # 从 kwargs 中弹出 "zero_size" 参数，默认为 False
        zero_size = kwargs.pop("zero_size", False)
        
        # 如果正在使用 CUDA，Kineto 可用，并且 CUDA 是支持的分析活动之一
        if (
            is_cuda
            and torch.autograd.kineto_available()
            and torch.profiler.ProfilerActivity.CUDA
            in torch.profiler.supported_activities()
        ):
            # 使用 torch.profiler 进行性能分析
            with torch.profiler.profile() as p:
                # 调用 self.func 方法执行具体功能，传入 inputs 和 kwargs
                actual = self.func(*inputs, **kwargs)
            # 获取性能分析结果的关键指标
            keys = tuple([e.key for e in p.key_averages()])
            # 检查是否调用了 multi_tensor_apply_kernel 函数
            mta_called = any("multi_tensor_apply_kernel" in k for k in keys)
            # 断言是否符合预期的快速路径和非零尺寸条件
            assert mta_called == (
                expect_fastpath and (not zero_size)
            ), f"{mta_called=}, {expect_fastpath=}, {zero_size=}, {self.func.__name__=}, {keys=}"
        else:
            # 若不满足性能分析条件，则直接调用 self.func 方法
            actual = self.func(*inputs, **kwargs)
        
        # 如果是原地操作，确保输入对象的 ID 等于输出对象的 ID
        if self.is_inplace:
            assert id(inputs[0]) == id(actual)
        
        # 返回实际处理后的结果
        return actual
# 创建一个类 InplaceForeachVersionBumpCheck，用于检查 inplace 操作的版本增加情况
class InplaceForeachVersionBumpCheck:
    def __init__(
        self, testcase: TestCase, tensorlist: "List[torch.Tensor]"  # noqa: F821
    ) -> None:
        # 初始化方法，接受一个 TestCase 实例和一个 torch.Tensor 列表作为输入参数
        self._testcase = testcase
        self._tensorlist = tensorlist
        # 保存原始 tensorlist 中各张量的版本号列表
        self._orig_version_counts = [t._version for t in tensorlist]

    # 定义 __enter__ 方法，用于上下文管理器，这里暂不做任何操作
    def __enter__(self):
        pass

    # 定义 __exit__ 方法，用于上下文管理器，用于检查执行过程中版本号是否有增加
    def __exit__(self, exc_type, exc_value, traceback):
        # 断言当前各张量的版本号都大于等于执行前的版本号
        # 注意：这里使用了 TestCase 的 assertGreaterEqual 方法来进行断言
        self._testcase.assertGreaterEqual(
            [t._version for t in self._tensorlist], self._orig_version_counts
        )


# 定义一个函数 get_transform_func，返回一个转换函数 transform
def get_transform_func(num_tensors, dtype, device, is_fastpath):
    # 定义内部函数 transform，根据输入的张量 t 进行转换或创建新张量
    def transform(t):
        if not torch.is_tensor(t):  # 如果 t 不是张量，则直接返回 t
            return t
        if torch.is_tensor(t) and t.ndim == 0:  # 如果 t 是零维张量，则直接返回 t
            return t
        # 否则，根据输入参数创建一个新的张量，并返回
        return make_tensor(
            (num_tensors, num_tensors),
            dtype=dtype,
            device=device,
            requires_grad=True,
            noncontiguous=not is_fastpath,
        )

    return transform  # 返回内部函数 transform


# 使用 unittest.mock.patch.dict 对 os.environ 进行字典更新，设置 KINETO_LOG_LEVEL 为 "5"
@unittest.mock.patch.dict(os.environ, {"KINETO_LOG_LEVEL": "5"})
# 定义一个测试类 TestForeach，继承自 TestCase
class TestForeach(TestCase):
    # 定义一个属性方法 is_cuda，用于判断当前设备是否为 CUDA
    @property
    def is_cuda(self):
        return self.device_type == "cuda"

    # 定义一个方法 _get_funcs，根据操作 op 返回一组函数包装器
    def _get_funcs(self, op):
        return (
            ForeachFuncWrapper(op.method_variant),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant),
            RegularFuncWrapper(op.ref_inplace),
        )

    # note(crcrpar): 确保零大小张量在 multi_tensor_apply 中得到适当的忽略
    # 这个问题最初在 https://github.com/pytorch/pytorch/issues/94865 中被报告
    # 相关链接：
    #   - https://github.com/pytorch/pytorch/pull/94655
    #   - https://github.com/pytorch/pytorch/issues/100701
    #   - https://github.com/pytorch/pytorch/pull/100811
    @onlyCUDA  # 使用装饰器 onlyCUDA，限定仅在 CUDA 上运行的测试
    @ops(  # 使用 ops 装饰器声明一系列操作，包括一元、二元、逐元素等操作
        foreach_unary_op_db
        + foreach_binary_op_db
        + foreach_pointwise_op_db
        + foreach_reduce_op_db
        + foreach_other_op_db,
        dtypes=(torch.float32,),  # 操作的数据类型限定为 torch.float32
    )
    # 定义一个测试方法，用于验证所有零大小张量不会触发内核启动
    def test_all_zero_size_tensors_do_not_launch_kernel(self, device, dtype, op):
        # 获取操作函数及其变种
        wrapped_op, _, inplace_op, _ = self._get_funcs(op)

        # 遍历零大小输入样本集合
        for sample in op.sample_zero_size_inputs(device, dtype):
            # 如果操作有方法变体，则调用包装的操作函数
            if op.method_variant is not None:
                wrapped_op(
                    (sample.input, *sample.args),
                    is_cuda=self.is_cuda,
                    expect_fastpath=True,
                    zero_size=True,
                )

            # 如果操作有原位变体，则使用原位操作函数进行调用
            if op.inplace_variant is not None:
                # 使用 InplaceForeachVersionBumpCheck 上下文管理器检查版本变化
                with InplaceForeachVersionBumpCheck(self, sample.input):
                    inplace_op(
                        (sample.input, *sample.args),
                        is_cuda=self.is_cuda,
                        expect_fastpath=True,
                        zero_size=True,
                    )

    # 如果 ROCm 版本小于 6.0，则跳过此测试
    @skipIfRocmVersionLessThan((6, 0))
    # 使用 ops 装饰器，指定测试操作集合，包括一元、二元、逐点和归约操作
    @ops(
        foreach_unary_op_db
        + foreach_binary_op_db
        + foreach_pointwise_op_db
        + foreach_reduce_op_db
        + foreach_other_op_db,
    )
    # 参数化测试，验证非连续内存和原位操作的不同组合
    @parametrize(
        "noncontiguous,inplace",
        [(False, False), (False, True), (True, False), (True, True)],
        # 自定义测试名称函数，根据参数生成测试名称
        name_fn=lambda x, y: "{}_{}".format(
            "fastpath" if not x else "slowpath", "inplace" if y else "outplace"
        ),
    )
    # 如果 CUDA 可用且设备不是 SM86 架构，则跳过测试
    @unittest.skipIf(
        torch.cuda.is_available() and not torch.cuda.get_device_capability(0) == (8, 6),
        "failing flakily on non sm86 cuda jobs",
    )
    # 定义一个测试函数，用于测试某种操作的奇偶性，接受多个参数
    def test_parity(self, device, dtype, op, noncontiguous, inplace):
        # 根据 inplace 参数确定要使用的函数和参考函数
        if inplace:
            _, _, func, ref = self._get_funcs(op)
        else:
            func, ref, _, _ = self._get_funcs(op)
        # 遍历操作的样本输入
        for sample in op.sample_inputs(device, dtype, noncontiguous=noncontiguous):
            ref_kwargs = sample.kwargs
            # 对于除法操作，如果涉及整数或布尔类型并且操作名是 "_foreach_div"，则不使用快速路径
            div_slowpath = (
                dtype in integral_types_and(torch.bool) and op.name == "_foreach_div"
            )
            # 预期是否可以使用快速路径取决于非连续数组、禁用快速路径或者除法操作的慢路径
            expect_fastpath = not (
                noncontiguous or sample.disable_fastpath or div_slowpath
            )
            ref_input, ctxmgr = sample.input, nullcontext()
            # 如果是 inplace 操作，则使用 torch.no_grad() 来执行深拷贝和分离操作
            if inplace:
                with torch.no_grad():
                    ref_input = [t.clone().detach() for t in sample.input]
                # 使用 InplaceForeachVersionBumpCheck 上下文管理器
                ctxmgr = InplaceForeachVersionBumpCheck(self, sample.input)
            try:
                with ctxmgr:
                    # 执行操作函数，传入样本输入和其他参数
                    actual = func(
                        [sample.input, *sample.args],
                        self.is_cuda,
                        expect_fastpath,
                        **sample.kwargs,
                    )
            except Exception as e:
                with (
                    # 捕获并断言预期的异常
                    self.assertRaisesRegex(type(e), re.escape(str(e)))
                    if not (op.has_no_in_place or not op.supports_out)
                    else self.assertRaises(type(e))
                ):
                    # 调用参考函数处理参考输入和参数
                    ref([ref_input, *sample.ref_args], **ref_kwargs)
            else:
                # 执行预期的结果，并断言实际结果与预期结果相等
                expected = ref([ref_input, *sample.ref_args], **ref_kwargs)
                self.assertEqual(expected, actual)

    # 定义一个私有方法用于执行二进制测试，接受多个参数
    def _binary_test(
        self,
        dtype,
        op,
        ref,
        inputs,
        is_fastpath,
        is_inplace,
        *,
        alpha,
        scalar_self_arg: bool,
    ):
        # 如果操作支持标量 self 参数，则为真，否则为假
        ref_inputs = (
            [[t.clone().detach() for t in inputs[0]], inputs[1]]
            if is_inplace
            else inputs
        )
        # 如果是原地操作，则将输入张量的克隆分离后的列表与第二个输入作为参考输入
        # 否则直接使用输入作为参考输入
        try:
            # 如果操作是原地操作，则使用 InplaceForeachVersionBumpCheck 进行版本检查
            with InplaceForeachVersionBumpCheck(
                self, inputs[0]
            ) if op.is_inplace else nullcontext():
                # 执行操作，获取实际输出
                actual = op(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            # 如果运行时发生异常，则断言异常类型和异常消息
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                # 如果没有标量 self 参数，则对参考输入进行参考函数调用
                if not scalar_self_arg:
                    ref(ref_inputs)
                else:
                    # 如果有标量 self 参数，则对每个输入执行参考函数调用
                    [ref.func(ref_inputs[0], t) for t in ref_inputs[1]]
        else:
            # 如果没有异常，则计算预期输出
            expected = (
                ref(ref_inputs)
                if not scalar_self_arg
                else [ref.func(ref_inputs[0], t) for t in ref_inputs[1]]
            )
            # 断言实际输出与预期输出相等
            self.assertEqual(actual, expected)
        # 如果 alpha 不为 None 并且没有标量 self 参数，则设置 kwargs 字典并更新输入为参考输入
        if alpha is not None and not scalar_self_arg:
            kwargs = {"alpha": alpha}
            ref_inputs = inputs
            try:
                # 创建操作参数字典并执行操作，获取实际输出
                op_kwargs = {}
                op_kwargs.update(kwargs)
                with InplaceForeachVersionBumpCheck(
                    self, inputs[0]
                ) if op.is_inplace else nullcontext():
                    actual = op(inputs, self.is_cuda, is_fastpath, **op_kwargs)
            except RuntimeError as e:
                # 如果运行时发生异常，则断言异常类型和异常消息
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    # 对参考输入和参数 kwargs 执行参考函数调用
                    ref(ref_inputs, **kwargs)
            else:
                # 如果没有异常，则计算预期输出
                expected = ref(ref_inputs, **kwargs)
                # 如果数据类型是 torch.float16 或 torch.bfloat16 并且测试使用 ROCM，则使用指定的容差进行断言
                if dtype in (torch.float16, torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(
                        expected, actual, atol=1.0e-3, rtol=default_tolerances(dtype)[0]
                    )
                else:
                    # 否则，断言实际输出与预期输出相等
                    self.assertEqual(expected, actual)
    # 定义测试函数，测试与标量自身支持的二元操作
    def test_binary_op_with_scalar_self_support(self, device, dtype, op, is_fastpath):
        
        # 定义递归克隆函数，用于克隆复杂对象
        def clone(arg):
            # 如果参数是列表或元组，递归克隆其中的每个元素
            if isinstance(arg, (list, tuple)):
                return [clone(a) for a in arg]
            # 如果参数是张量，则克隆该张量并设置其需要梯度
            if torch.is_tensor(arg):
                return arg.clone().detach().requires_grad_()
            else:
                return arg
        
        # 标志：标量自身参数测试是否完成
        scalar_self_arg_test_complete = False
        
        # 遍历操作对象提供的样本输入
        for i, sample in enumerate(
            op.sample_inputs(device, dtype, noncontiguous=not is_fastpath)
        ):
            # 解包样本参数
            (rhs_arg,) = sample.args
            
            # 准备关键字参数，如果有alpha参数，则从kwargs中弹出
            kwargs = {} or sample.kwargs
            alpha = kwargs.pop("alpha", None)
            
            # 获取操作函数及其引用函数
            wrapped_op, ref, inplace_op, inplace_ref = self._get_funcs(op)
            
            # 如果rhs_arg是数值且标量自身参数测试未完成，则执行标量自身参数测试
            if isinstance(rhs_arg, Number) and not scalar_self_arg_test_complete:
                scalar_self_arg_test_complete = True
                
                # 执行二元测试，测试操作是否正确处理标量自身作为参数的情况
                self._binary_test(
                    dtype,
                    wrapped_op,
                    ref,
                    [rhs_arg, sample.input],
                    is_fastpath,
                    False,
                    alpha=alpha,
                    scalar_self_arg=True,
                )
                
                # 如果操作支持自动求导且dtype为torch.float32
                if op.supports_autograd and dtype == torch.float32:
                    # 对样本进行变换，获取变换后的输入张量
                    transformed_sample = sample.transform(
                        get_transform_func(
                            len(sample.input), dtype, device, is_fastpath
                        )
                    )
                    tensors = transformed_sample.input
                    (rhs_arg,) = transformed_sample.args
                    
                    # 克隆输入张量及rhs_arg
                    ref_tensors, ref_rhs_arg = clone(tensors), clone(rhs_arg)
                    
                    # 计算操作后的梯度并进行反向传播
                    sum(
                        wrapped_op(
                            [rhs_arg, tensors], is_cuda=False, expect_fastpath=False
                        )
                    ).mean().backward()
                    
                    # 计算参考函数后的梯度并进行反向传播
                    sum(ref.func(ref_rhs_arg, t) for t in ref_tensors).mean().backward()
                    
                    # 断言：检查梯度是否正确计算
                    self.assertEqual(
                        [t.grad for t in tensors], [t.grad for t in ref_tensors]
                    )

    # 使用@ops和@parametrize装饰器，对pointwise操作进行测试
    @ops(foreach_pointwise_op_db)
    @parametrize("is_fastpath", (True, False))
    def test_pointwise_op_with_tensor_of_scalarlist_overload(
        self, device, dtype, op, is_fastpath
    ):
    
        # 定义点操作测试函数，测试点操作函数
        def _pointwise_test(
            self,
            op,
            ref,
            inputs,
            is_fastpath,
            is_inplace,
            *,
            scalars=None,
            custom_values_err=None,
            **kwargs,
        ):
    ):
        # 准备参考输入数据，如果是原地操作，则对第一个输入列表进行克隆和分离，否则直接使用输入
        ref_inputs = (
            [[t.clone().detach() for t in inputs[0]], inputs[1], inputs[2]]
            if is_inplace
            else inputs
        )
        try:
            # 如果是原地操作，使用 InplaceForeachVersionBumpCheck 上下文管理器来检查版本增加情况
            # 否则使用 nullcontext() 上下文管理器
            with (
                InplaceForeachVersionBumpCheck(self, inputs[0])
                if is_inplace
                else nullcontext()
            ):
                # 调用操作函数 op 处理输入数据，获取实际输出
                actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
        except RuntimeError as e:
            # 捕获运行时异常，使用 self.assertRaisesRegex 断言捕获到的异常类型和错误消息
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(ref_inputs, **kwargs)
        else:
            # 没有异常发生时，计算期望输出，并使用 self.assertEqual 断言实际输出与期望输出相等
            expected = ref(ref_inputs, **kwargs)
            self.assertEqual(expected, actual)
        
        # 如果有标量值，则更新 kwargs，并重新尝试操作
        if scalars is not None:
            kwargs = kwargs.copy()
            kwargs["scalars"] = scalars
            try:
                # 再次调用操作函数 op 处理输入数据，获取实际输出
                actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
            except RuntimeError as e:
                # 捕获运行时异常，如果没有提供自定义错误消息，使用 self.assertRaisesRegex 断言捕获到的异常类型和错误消息
                if custom_values_err is None:
                    with self.assertRaisesRegex(type(e), re.escape(str(e))):
                        ref(ref_inputs, **kwargs)
                else:
                    # 否则，使用 self.assertEqual 断言异常消息与自定义消息相等
                    self.assertEqual(re.escape(str(e)), re.escape(custom_values_err))
            else:
                # 没有异常发生时，计算期望输出，并使用 self.assertEqual 断言实际输出与期望输出相等
                expected = ref(ref_inputs, **kwargs)
                self.assertEqual(expected, actual)
    # 使用装饰器 ops，传入一个 lambda 表达式过滤支持输出的操作，并使用给定的二进制操作数据库
    @ops(
        filter(lambda op: op.supports_out, foreach_binary_op_db),
        allowed_dtypes=[torch.float],
    )
    # 定义测试方法 test_binary_op_scalar_with_different_tensor_dtypes，接受设备、数据类型和操作参数
    def test_binary_op_scalar_with_different_tensor_dtypes(self, device, dtype, op):
        # 获取当前操作的方法变体
        foreach_op = op.method_variant
        # 创建包含两个张量的列表，一个是 float 类型，另一个是 long 类型，均在指定设备上
        tensors = [
            torch.tensor([1.1], dtype=torch.float, device=device),
            torch.tensor([1], dtype=torch.long, device=device),
        ]
        runtime_error = None
        try:
            # 尝试调用 foreach_op 方法，传入张量列表和标量参数 1
            foreach_op(tensors, 1)
        except RuntimeError as e:
            # 捕获 RuntimeError 异常
            runtime_error = e
        # 断言 runtime_error 为 None
        self.assertIsNone(runtime_error)

    # 使用装饰器 skipIfTorchDynamo，跳过由 Torch Dynamo 引起的测试
    @skipIfTorchDynamo("Different error msgs, TODO")
    # 使用 ops 装饰器，传入 lambda 表达式过滤支持输出的操作，并使用给定的二进制操作数据库和 OpDTypes.supported 数据类型
    @ops(
        filter(lambda op: op.supports_out, foreach_binary_op_db),
        dtypes=OpDTypes.supported,
    )
    # 使用 unittest.skipIf 装饰器，如果 CUDA 不可用则跳过测试
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    # 使用 ops 装饰器，传入 lambda 表达式过滤支持输出的操作，并使用给定的二进制操作数据库和 OpDTypes.supported 数据类型
    @ops(
        filter(lambda op: op.supports_out, foreach_binary_op_db),
        dtypes=OpDTypes.supported,
    )
    # 使用 unittest.skipIf 装饰器，如果 CUDA 设备不是 sm86 则跳过测试
    @unittest.skipIf(
        torch.cuda.is_available() and not torch.cuda.get_device_capability(0) == (8, 6),
        "failing flakily on non sm86 cuda jobs, ex https://github.com/pytorch/pytorch/issues/125775",
    )
    # 使用 ops 装饰器，传入 lambda 表达式过滤支持输出的操作，并使用给定的二进制操作数据库和半精度浮点类型
    @ops(
        filter(lambda op: op.supports_out, foreach_binary_op_db),
        dtypes=floating_types_and(torch.half, torch.bfloat16),
    )
    # 使用 unittest.skipIf 装饰器，如果 CUDA 设备不是 sm86 则跳过测试
    @unittest.skipIf(
        torch.cuda.is_available() and not torch.cuda.get_device_capability(0) == (8, 6),
        "failing flakily on non sm86 cuda jobs",
    )
    # 定义测试方法 test_binary_op_float_inf_nan，接受设备、数据类型和操作参数
    def test_binary_op_float_inf_nan(self, device, dtype, op):
        # 创建包含两个列表的元组 inputs，每个列表包含特定的浮点无穷大和 NaN 值的张量，均在指定设备上
        inputs = (
            [
                torch.tensor([float("inf")], device=device, dtype=dtype),
                torch.tensor([-float("inf")], device=device, dtype=dtype),
                torch.tensor([float("nan")], device=device, dtype=dtype),
                torch.tensor([float("nan")], device=device, dtype=dtype),
            ],
            [
                torch.tensor([-float("inf")], device=device, dtype=dtype),
                torch.tensor([float("inf")], device=device, dtype=dtype),
                torch.tensor([float("inf")], device=device, dtype=dtype),
                torch.tensor([float("nan")], device=device, dtype=dtype),
            ],
        )
        # 获取 op 的函数引用、参考函数引用、就地操作函数引用和就地参考函数引用
        op, ref, inplace_op, inplace_ref = self._get_funcs(op)
        # 调用 _binary_test 方法，测试操作函数 op，参考函数 ref，传入 inputs 等参数
        self._binary_test(
            dtype, op, ref, inputs, True, False, alpha=None, scalar_self_arg=False
        )
        # 调用 _binary_test 方法，测试就地操作函数 inplace_op，就地参考函数 inplace_ref，传入 inputs 等参数
        self._binary_test(
            dtype,
            inplace_op,
            inplace_ref,
            inputs,
            True,
            True,
            alpha=None,
            scalar_self_arg=False,
        )

    # 注释：以下三个测试用例（后缀为 `_tensors_on_different_devices`）
    # 检查 foreach 是否能够处理张量列表位于不同设备但索引相同的情况，例如 ['cuda', 'cpu']
    @onlyCUDA
    # 使用 ops 装饰器，传入 foreach_unary_op_db 的操作过滤器
    @ops(foreach_unary_op_db)
    # 测试不同设备上的一元操作张量
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        # 获取操作的方法和参考函数
        method, ref, inplace_method, ref_inplace = self._get_funcs(op)
        
        # 从操作的示例输入中获取张量，这些张量包括指定设备和数据类型的多个输入张量
        tensors = next(
            iter(op.sample_inputs(device, dtype, num_input_tensors=[2]))
        ).input
        
        # 将第二个张量移动到CPU设备
        tensors[1] = tensors[1].to("cpu")
        
        # 如果操作不支持输出张量
        if not op.supports_out:
            try:
                # 调用方法执行操作，此处不使用原地操作，不生成零大小的张量
                actual = method((tensors,), False, False, zero_size=False)
            except RuntimeError as e:
                # 捕获运行时异常并断言引发的异常与预期的异常类型和消息相匹配
                with self.assertRaisesRegex(type(e), str(e)):
                    ref((tensors,))
            else:
                # 获取参考实现的期望结果
                expected = ref((tensors,))
                # 断言实际结果与期望结果相等
                self.assertEqual(expected, actual)
        
        try:
            # 执行原地方法操作，此处不使用原地操作，不生成零大小的张量
            inplace_method((tensors,), False, False, zero_size=False)
        except RuntimeError as e:
            # 捕获运行时异常并断言引发的异常与预期的异常类型和消息相匹配
            with self.assertRaisesRegex(type(e), str(e)):
                ref_inplace((tensors,))
        else:
            # 如果操作不支持输出张量，断言实际结果与张量本身相等，否则与期望结果相等
            if not op.supports_out:
                self.assertEqual(expected, tensors)
            else:
                self.assertEqual([torch.zeros_like(t) for t in tensors], tensors)

    # 仅在CUDA环境下执行的测试函数
    @onlyCUDA
    # 使用支持输出的二元操作函数和给定的二元操作数据库
    @ops(filter(lambda op: op.supports_out, foreach_binary_op_db))
    # 在满足特定条件时跳过测试，这里检查CUDA是否可用且设备兼容性不符合特定版本
    @unittest.skipIf(
        torch.cuda.is_available() and not torch.cuda.get_device_capability(0) == (8, 6),
        "failing flakily on non sm86 cuda jobs",
    )
    # 测试不同设备上的二元操作张量
    def test_binary_op_tensors_on_different_devices(self, device, dtype, op):
        # 从操作的示例输入中获取CUDA和CPU设备上的相同大小的输入张量
        _cuda_tensors = next(
            iter(op.sample_inputs(device, dtype, num_input_tensors=[2], same_size=True))
        ).input
        _cpu_tensors = next(
            iter(op.sample_inputs("cpu", dtype, num_input_tensors=[2], same_size=True))
        ).input
        
        # 将CUDA和CPU张量分别解压缩为两个列表
        tensors1, tensors2 = list(zip(_cuda_tensors, _cpu_tensors))

        # 获取操作的方法和参考函数的变体
        foreach_op, foreach_op_ = op.method_variant, op.inplace_variant
        native_op, native_op_ = op.ref, op.ref_inplace
        
        try:
            # 执行张量的元素级foreach操作
            actual = foreach_op(tensors1, tensors2)
        except RuntimeError as e:
            # 捕获运行时异常并断言引发的异常与预期的异常类型和消息相匹配
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            # 获取参考实现的期望结果
            expected = [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
            # 断言实际结果与期望结果相等
            self.assertEqual(expected, actual)
        
        try:
            # 执行张量的原地操作
            foreach_op_(tensors1, tensors2)
        except RuntimeError as e:
            # 捕获运行时异常并断言引发的异常与预期的异常类型和消息相匹配
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op_(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            # 断言实际结果与第一组张量相等
            self.assertEqual(actual, tensors1)

    # 仅在CUDA环境下执行的测试函数
    @onlyCUDA
    # 使用点对点操作数据库和浮点数数据类型执行操作
    @ops(foreach_pointwise_op_db, allowed_dtypes=floating_types())
    # 定义测试方法，用于在不同设备上测试张量的逐点操作
    def test_pointwise_op_tensors_on_different_devices(self, device, dtype, op):
        # tensors1: 在 'cuda' 和 'cpu' 设备上的张量列表
        # tensors2: 在 'cuda' 和 'cpu' 设备上的张量列表
        # tensors3: 在 'cuda' 和 'cpu' 设备上的张量列表
        # 当 dtype 是 torch.float32 时，第一个张量列表是零尺寸的
        _cuda_tensors = list(
            op.sample_inputs(device, dtype, num_input_tensors=[3], same_size=True)
        )[int(dtype == torch.float32)].input
        _cpu_tensors = next(
            iter(op.sample_inputs("cpu", dtype, num_input_tensors=[3], same_size=True))
        ).input
        tensors1, tensors2, tensors3 = list(zip(_cuda_tensors, _cpu_tensors))

        # 获取操作的三个变种函数
        foreach_op, foreach_op_, native_op = (
            op.method_variant,
            op.inplace_variant,
            op.ref,
        )
        # 使用 foreach_op 对 tensors1、tensors2 和 tensors3 进行操作
        actual = foreach_op(tensors1, tensors2, tensors3)
        # 计算期望的结果，分别基于 _cuda_tensors 和 _cpu_tensors
        expected = [native_op(*_cuda_tensors), native_op(*_cpu_tensors)]
        self.assertEqual(expected, actual)

        # 注意(mkozuki): 限制数据类型为 FP32 和 FP64，可以安全地运行原位操作。
        # 使用 foreach_op_ 对 tensors1、tensors2 和 tensors3 进行原位操作
        foreach_op_(tensors1, tensors2, tensors3)
        self.assertEqual(expected, tensors1)

    # 注释: BFloat16 与 FP32 具有相同数量的指数位，因此如果 BF16 中的平方 L2 范数溢出，则 FP32 也会溢出。
    @onlyCUDA
    @ops(
        [o for o in foreach_reduce_op_db if "norm" in o.name],
        allowed_dtypes=(torch.half, torch.bfloat16),
    )
    # 定义测试方法，用于测试在大数值输入时的逐元素 L2 范数计算
    def test_foreach_l2_large_value_input(self, device, dtype, op):
        ord, N = 2, 10
        # 获取指定数据类型的最大值
        max_value = torch.finfo(dtype).max
        # 创建一个张量，其值为 max_value 的平方根，并将其移到指定设备上
        scaler = torch.tensor([max_value]).sqrt().to(device=device, dtype=dtype)
        # 生成输入张量列表，每个张量都乘以 scaler，并保证梯度计算开启
        inputs = (
            [
                t * scaler
                for t in next(
                    iter(
                        op.sample_inputs(
                            device,
                            dtype,
                            requires_grad=True,
                            num_input_tensors=[N],
                            low=1,
                        )
                    )
                ).input
            ][:-1],
        )
        # 确保每个张量的平方 L2 范数的最小值大于 dtype 的最大值
        self.assertTrue(scaler * scaler * N > max_value)
        # 获取操作函数和参考函数
        fn, ref_fn, *_ = self._get_funcs(op)
        # 使用 fn 计算实际的 L2 范数，指定在 CUDA 上运行，期望快速路径，并不考虑零尺寸情况
        actual = fn(
            inputs, is_cuda=True, expect_fastpath=True, ord=ord, zero_size=False
        )
        # 计算期望的 L2 范数
        expect = ref_fn(inputs, ord=ord)

        if dtype == torch.float16:
            # 确保参考的 L2 范数值在 FP16 范围内
            self.assertFalse(any(torch.isinf(e) for e in expect))
        else:
            # 确保所有输入的零尺寸情况下，参考的 L2 范数值为无穷大或对应张量元素个数为零
            self.assertTrue(
                all(
                    inputs[0][i].numel() == 0 or torch.isinf(e)
                    for i, e in enumerate(expect)
                )
            )
        # 断言实际计算的 L2 范数与期望值相等，忽略 NaN 值的比较
        self.assertEqual(expect, actual, equal_nan=False)

    @onlyCUDA
    @ops(foreach_reduce_op_db, allowed_dtypes=floating_types())
    # 注释结束
    # 使用 pytest 的 parametrize 装饰器，为 test_big_num_tensors 方法参数 use_cuda_graph 添加参数组 (False, True)，分别测试不同条件下的情况
    @parametrize("use_cuda_graph", (False, True))
    # 定义测试方法 test_big_num_tensors，接受 device, dtype, op, use_cuda_graph 四个参数
    def test_big_num_tensors(self, device, dtype, op, use_cuda_graph):
        # 定义 tensorlist，包含 N 个形状为 (2, 3) 的张量，数据类型为 dtype，存储于 device 上，要求是非连续的张量
        N = 600
        tensorlist = [
            make_tensor((2, 3), dtype=dtype, device=device, noncontiguous=False)
            for _ in range(N)
        ]
        # 获取操作函数 fn 和参考函数 ref_fn
        fn, ref_fn, *_ = self._get_funcs(op)

        import math

        # 根据操作的名称设置 ords 变量，用于指定计算范数的阶数
        if op.name == "_foreach_norm":
            ords = (1, 2, math.inf)
        else:
            ords = (None,)

        # 遍历 ords 中的每个 ord
        for ord in ords:
            # 根据 ord 的值，创建 kwargs 字典，用于传递给 fn 和 ref_fn 函数
            kwargs = {"ord": ord} if ord else {}
            # 根据 use_cuda_graph 的值选择不同的计算路径
            if not use_cuda_graph:
                # 如果不使用 CUDA 图，调用 fn 函数计算张量列表的结果
                actual = fn(
                    inputs=[tensorlist],
                    is_cuda=True,
                    expect_fastpath=True,
                    zero_size=False,
                    **kwargs,
                )
            else:
                # 当使用 CUDA 图并且张量元数据不适合静态内核参数空间时，multi_tensor_apply 会创建 launch 参数一次，
                # 使用 cudaUserObject_t 将其生命周期绑定到图中，并在重播过程中重复使用。此测试验证 multi_tensor_apply 在此场景下的行为。
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    # 在 CUDA 图上执行 fn.func 函数计算张量列表的结果
                    actual = fn.func(tensorlist, **kwargs)
                g.replay()
            # 使用 ref_fn 函数计算参考结果
            expect = ref_fn(inputs=[tensorlist], **kwargs)

            # 使用 self.assertEqual 断言验证实际结果和期望结果的相等性，允许 NaN 相等
            self.assertEqual(expect, actual, equal_nan=True)

    # 使用 onlyCUDA 装饰器，确保该测试仅在 CUDA 环境下运行
    @onlyCUDA
    # 使用 ops 装饰器，指定测试使用 foreach_reduce_op_db 中的操作
    @ops(foreach_reduce_op_db)
    # 定义测试方法 test_foreach_reduce_large_input，接受 device, dtype, op 三个参数
    def test_foreach_reduce_large_input(self, device, dtype, op):
        # 测试输入的大小超过 kChunkSize = 65536
        N = 65536 * 2
        disable_fastpath = False
        kwargs = {}
        # 根据操作的名称设置 ord 变量，并检查是否需要禁用快速路径
        if op.name == "_foreach_norm":
            ord = 2
            disable_fastpath = not (
                ord in (1, 2)
                and dtype in floating_types_and(torch.half, torch.bfloat16)
            )
            kwargs["ord"] = ord

        # 创建输入张量列表，包含一个形状为 (N,) 的张量
        inputs = ([make_tensor((N,), dtype=dtype, device=device, noncontiguous=False)],)
        # 获取包装后的操作函数 wrapped_op 和参考函数 ref
        wrapped_op, ref, _, _ = self._get_funcs(op)
        # 使用 self.assertEqual 断言验证 ref 函数和 wrapped_op 函数计算结果的相等性
        self.assertEqual(
            ref(inputs, **kwargs),
            wrapped_op(
                inputs, self.is_cuda, not disable_fastpath, zero_size=False, **kwargs
            ),
        )

    # 使用 onlyCUDA 装饰器，确保该测试仅在 CUDA 环境下运行
    @onlyCUDA
    # 使用 ops 装饰器，指定测试使用 foreach_unary_op_db、foreach_binary_op_db、foreach_pointwise_op_db、foreach_other_op_db 中的操作，
    # 且数据类型为 torch.float
    @ops(
        foreach_unary_op_db
        + foreach_binary_op_db
        + foreach_pointwise_op_db
        + foreach_other_op_db,
        dtypes=(torch.float,),
    )
    # 测试函数，用于检查原位操作的叶子变量的梯度函数
    def test_inplace_foreach_leaf_check_and_grad_fn(self, device, dtype, op):
        # 获取原位操作的原地变体
        inplace_op = op.inplace_variant
        # 如果没有原位变体，则跳过此测试
        if inplace_op is None:
            self.skipTest("no in-place op available")

        # 获取一个样本输入，这里使用操作的样本输入生成器
        sample = next(
            iter(
                op.sample_inputs(
                    dtype=dtype, device=device, num_input_tensors=[2], same_size=True
                )
            )
        )
        # 将第一个输入的 requires_grad 属性设为 True，用于测试梯度
        sample.input[0].requires_grad_(True)
        # 使用断言检查是否会抛出 RuntimeError，要求叶子变量需要梯度
        with self.assertRaisesRegex(RuntimeError, "a leaf Variable that requires grad"):
            inplace_op(sample.input, *sample.args)
        # 将第二个输入的 requires_grad 属性设为 True，用于测试梯度
        sample.input[1].requires_grad_(True)
        # 使用断言再次检查是否会抛出 RuntimeError，要求叶子变量需要梯度
        with self.assertRaisesRegex(RuntimeError, "a leaf Variable that requires grad"):
            inplace_op(sample.input, *sample.args)

        # 对样本输入的每个张量进行克隆、分离、并设置是否需要梯度的操作
        _tensors = [
            t.clone().detach().requires_grad_(i == 0)
            for i, t in enumerate(sample.input)
        ]
        # 对克隆的张量进行进位操作
        tensors = [t.clone() for t in _tensors]
        inplace_op(tensors, *sample.args)
        # 使用断言验证第一个张量是否有梯度函数
        self.assertIsNotNone(tensors[0].grad_fn)
        # 使用断言验证第二个张量是否没有梯度函数
        self.assertIsNone(tensors[1].grad_fn)

    # 标记为只有 CUDA 环境下运行的测试函数
    @onlyCUDA
    # 使用指定操作函数进行测试
    @ops(
        # 过滤出支持输出的操作函数
        filter(
            lambda op: op.supports_out,
            # 汇总所有数据库中的操作函数
            foreach_unary_op_db
            + foreach_binary_op_db
            + foreach_pointwise_op_db
            + foreach_other_op_db,
        ),
        # 指定数据类型为 torch.float 类型
        dtypes=(torch.float,),
    )
    # 测试带有无效梯度的输出函数
    def test_outplace_with_invalid_grads(self, device, dtype, op):
        # 获取操作的功能函数
        func, *_ = self._get_funcs(op)
        # 获取一个样本输入，这里使用操作的样本输入生成器
        sample = next(
            iter(
                op.sample_inputs(
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                    num_input_tensors=[2],
                    same_size=True,
                )
            )
        )
        # 使用断言验证样本输入的所有张量是否都需要梯度
        self.assertTrue(all(t.requires_grad for t in sample.input))
        # 调用功能函数计算输出
        (out1, out2) = func(
            [sample.input, *sample.args],
            is_cuda=False,
            expect_fastpath=False,
            **sample.kwargs,
        )
        # 对 out1 进行反向传播
        out1.backward(torch.ones_like(out1))
        # 使用断言验证第一个输入张量是否有梯度
        self.assertIsNotNone(sample.input[0].grad)
        # 使用断言验证第二个输入张量是否没有梯度
        self.assertIsNone(sample.input[1].grad)

    # 测试需要结果进行反向传播的操作函数
    @ops(
        # 过滤出需要结果进行反向传播的操作函数
        filter(
            lambda op: op.backward_requires_result,
            # 汇总所有数据库中的操作函数
            foreach_unary_op_db
            + foreach_binary_op_db
            + foreach_pointwise_op_db
            + foreach_other_op_db,
        ),
        # 指定数据类型为 torch.float32 类型
        dtypes=(torch.float32,),
    )
    # 定义测试函数，测试梯度函数的生命周期当结果被保存时
    def test_lifetime_of_grad_fn_when_result_is_saved(self, device, dtype, op):
        # 定义内部函数，用于获取参考对象
        def get_ref(func, sample):
            # 定义一个空的类 Foo
            class Foo:
                pass

            # 调用 func 函数计算结果，获取输出 out
            out = func(
                (sample.input, *sample.args),
                is_cuda=False,
                expect_fastpath=False,
                **sample.kwargs,
            )
            # 创建一个 Foo 类的实例 foo
            foo = Foo()
            # 获取 out 的第一个元素的梯度函数的元数据字典
            meta_dict = out[0].grad_fn.metadata
            # 将 foo 存储到 meta_dict 的第一个位置
            meta_dict[0] = foo
            # 创建 foo 的弱引用 ref
            ref = weakref.ref(foo)
            return out, ref

        # 定义内部测试函数，用于测试函数的生命周期
        def _test(func, sample):
            # 调用 get_ref 函数获取输出和参考对象的弱引用 ref
            out, ref = get_ref(func, sample)
            # 断言 ref 不为空
            self.assertIsNotNone(ref())
            # 删除 out 对象
            del out
            # 断言 ref 为空
            self.assertIsNone(ref())

        # 从 op 中获取函数 func
        func = self._get_funcs(op)[0]
        # 遍历 op 的样本输入
        for sample in op.sample_inputs(
            device, dtype, requires_grad=True, num_input_tensors=[1]
        ):
            # 删除 sample.kwargs 中的 "is_fastpath" 和 "disable_fastpath" 键
            for key in ("is_fastpath", "disable_fastpath"):
                if key in sample.kwargs:
                    del sample.kwargs[key]
            # 如果 op 的名称为 "_foreach_pow"，执行以下条件检查和跳过
            # 参考链接: https://github.com/pytorch/pytorch/blob/5403c777/tools/autograd/derivatives.yaml#L3048-L3049
            if op.name == "_foreach_pow":
                # 如果 sample.args[0] 是列表且第一个元素是数字，或者 sample.args[0] 是数字但不是浮点数，则跳过
                if (
                    isinstance(sample.args[0], list)
                    and isinstance(sample.args[0][0], Number)
                ) or (
                    isinstance(sample.args[0], Number)
                    and not isinstance(sample.args[0], float)
                ):
                    continue
                # 如果 sample.args[0] 是浮点数，则重新设置参数
                if isinstance(sample.args[0], float):
                    new_args = (sample.input,)
                    sample.input = sample.args[0]
                    sample.args = new_args
            # 执行 _test 函数进行测试
            _test(func, sample)

    # 如果不支持多 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # 测试张量分组函数
    def test_tensors_grouping(self):
        # 每个列表中张量的数量
        num_tensors_per_list = 10
        # CUDA 设备的数量
        num_devices = torch.cuda.device_count()
        # 张量的数据类型列表
        dtypes = (torch.float16, torch.float32, torch.float64)
        # 创建包含随机设备和数据类型的张量列表 list1
        list1 = [
            torch.tensor(
                i,
                device=torch.device("cuda", random.randint(0, num_devices - 1)),
                dtype=dtypes[random.randint(0, 2)],
            )
            for i in range(num_tensors_per_list)
        ]
        # 创建与 list1 相同长度的 None 列表 list2
        list2 = [None for _ in list1]
        # 创建与 list1 相同长度的随机张量列表 list3
        list3 = [torch.rand_like(t) for t in list1]
        # 嵌套张量列表 nested_tensorlists
        nested_tensorlists = [list1, list2, list3]
        # 使用 torch.utils._foreach_utils._group_tensors_by_device_and_dtype 函数对张量进行按设备和数据类型分组
        grouped_tensors = torch.utils._foreach_utils._group_tensors_by_device_and_dtype(
            nested_tensorlists, with_indices=True
        )
        # 已处理的张量数量
        num_tensors_seen = 0
        # 遍历分组后的张量 grouped_tensors
        for (device, dtype), ([l1, l2, l3], indices) in grouped_tensors.items():
            # 检查 l1 和 l3 中张量的设备和数据类型
            for t in itertools.chain(l1, l3):
                self.assertEqual(t.device, device)
                self.assertEqual(t.dtype, dtype)
                num_tensors_seen += 1
            # 检查 list1 和 list2 的长度是否相等
            self.assertEqual(len(l1), len(l2))
            # 检查 list2 中的元素是否全为 None
            self.assertTrue(all(p is None for p in l2))
            # 检查索引与原始列表元素的对应关系
            for i, index in enumerate(indices):
                self.assertEqual(l1[i], list1[index])
                self.assertEqual(l2[i], list2[index])
                self.assertEqual(l3[i], list3[index])
        # 检查处理的张量总数是否正确
        self.assertEqual(num_tensors_seen, 2 * num_tensors_per_list)

    # 仅适用于 CUDA 的测试装饰器
    @onlyCUDA
    def test_0dim_tensor_overload_cpu_ok(self):
        # 创建在 CUDA 设备上的零维张量列表
        tensors = [torch.ones((), device="cuda", dtype=torch.float32) for _ in range(2)]
        # 创建在 CPU 设备上的标量张量
        scalar_cpu_tensor = torch.tensor(4.0, device="cpu")

        # 对张量列表进行乘法操作，标量可以在 CPU 上
        actual = torch._foreach_mul(tensors, scalar_cpu_tensor)
        self.assertEqual(actual, [t.mul(scalar_cpu_tensor) for t in tensors])
        
        # 对张量列表进行除法操作，标量可以在 CPU 上
        actual = torch._foreach_div(tensors, scalar_cpu_tensor)
        self.assertEqual(actual, [t.div(scalar_cpu_tensor) for t in tensors])

    # 仅适用于 CUDA 的测试装饰器
    @onlyCUDA
    def test_div_reciprocal(self):
        # 期望的浮点数和指数部分
        expect_m, expect_e = torch.frexp(
            torch.div(torch.tensor(0.1, device="cuda"), 10.0)
        )
        # 实际的浮点数和指数部分
        actual_m, actual_e = torch.frexp(
            torch._foreach_div([torch.tensor(0.1, device="cuda")], [10.0])[0]
        )
        # 检查浮点数和指数部分是否匹配
        self.assertEqual(expect_m, actual_m)
        self.assertEqual(expect_e, actual_e)
    def test_0dim_tensor_overload_exception(self):
        # 检查快速路径的异常情况
        tensors = [
            make_tensor((2, 2), dtype=torch.float, device="cuda") for _ in range(2)
        ]
        # 断言在运行时出现特定异常信息
        with self.assertRaisesRegex(RuntimeError, "scalar tensor expected to be on"):
            # 调用 torch._foreach_add 函数，期望引发异常，传入 GPU 上的标量张量和 alpha 参数
            torch._foreach_add(tensors, torch.tensor(1.0, device="cpu"), alpha=1.0)

        tensors = [
            make_tensor((2, 2), dtype=torch.float, device=d) for d in ("cpu", "cuda")
        ]
        # 断言在运行时出现特定异常信息
        with self.assertRaisesRegex(
            RuntimeError, "scalar tensor expected to be 0 dim but"
        ):
            # 调用 torch._foreach_mul 函数，期望引发异常，传入 GPU 上的非零维张量
            torch._foreach_mul(tensors, torch.tensor([1.0, 1.0], device="cuda"))
        # 断言在运行时出现特定异常信息
        with self.assertRaisesRegex(
            RuntimeError, "scalar tensor expected to be 0 dim but"
        ):
            # 调用 torch._foreach_add 函数，期望引发异常，传入 GPU 上的非零维张量
            torch._foreach_add(tensors, torch.tensor([1.0, 1.0], device="cuda"))

    @onlyCUDA
    @ops(filter(lambda op: op.name == "_foreach_copy", foreach_binary_op_db))
    def test_foreach_copy_with_multi_device_inputs(self, device, dtype, op):
        # 获取 inplace_variant 和 ref_inplace 函数
        foreach_copy_ = op.inplace_variant
        copy_ = op.ref_inplace
        # 遍历非阻塞标志的两种情况
        for non_blocking in (False, True):
            # 遍历 op 的样本输入
            for sample in op.sample_inputs(device, dtype, noncontiguous=False):
                # 在没有梯度计算的上下文中
                with torch.no_grad():
                    # 克隆并分离样本输入的每个张量
                    ref_input = [t.clone().detach() for t in sample.input]
                # 调用 foreach_copy_ 函数，复制 sample.args[0] 到 sample.input 中，支持非阻塞操作
                foreach_copy_(sample.input, sample.args[0], non_blocking)
                # 逐个比较 ref_input 和 sample.args[0] 的元素
                for t, s in zip(ref_input, sample.args[0]):
                    # 使用 copy_ 函数将 s 复制到 t 中，支持非阻塞操作
                    copy_(t, s, non_blocking)
                # 断言 sample.input 等于 ref_input
                self.assertEqual(sample.input, ref_input)
                # 如果 CUDA 设备数大于 1
                if torch.cuda.device_count() > 1:
                    # 指定第二个 CUDA 设备
                    device = torch.device("cuda", 1)
                    # 将 sample.args[0] 的张量移动到 rhs_tensors
                    rhs_tensors = [t.to(device) for t in sample.args[0]]
                    # 调用 foreach_copy_ 函数，复制 rhs_tensors 到 sample.input 中，支持非阻塞操作
                    foreach_copy_(sample.input, rhs_tensors, non_blocking)
                    # 逐个比较 ref_input 和 rhs_tensors 的元素
                    for t, s in zip(ref_input, rhs_tensors):
                        # 使用 copy_ 函数将 s 复制到 t 中，支持非阻塞操作
                        copy_(t, s, non_blocking)
                    # 断言 ref_input 等于 sample.input
                    self.assertEqual(ref_input, sample.input)

    @onlyCUDA
    @ops(filter(lambda op: op.name == "_foreach_copy", foreach_binary_op_db))
    # 定义测试方法，用于验证在多种数据类型下执行 foreach 操作的复制行为
    def test_foreach_copy_with_multi_dtypes(self, device, dtype, op):
        # 检查以下条件：(a) 是否调用了 multi_tensor_apply，(b) 使用 for 循环和 Tensor.copy_ 方法时的数值一致性
        foreach_copy_ = ForeachFuncWrapper(op.inplace_variant)
        # 遍历操作的样本输入集合，其中包含指定设备和数据类型的样本数据
        for sample in op.sample_inputs(device, dtype, noncontiguous=False):
            # 遍历浮点类型和 torch.half、torch.bfloat16 类型的元组
            for src_dtype in floating_types_and(torch.half, torch.bfloat16):
                # 如果源数据类型与目标数据类型相同，则跳过当前循环
                if src_dtype == dtype:
                    continue
                # 复制样本输入中的每个 Tensor，并存储在 self_tensors 列表中
                self_tensors = [t.clone() for t in sample.input]
                # 将 self_tensors 中的每个 Tensor 转换为指定的源数据类型，并存储在 src_tensors 列表中
                src_tensors = [t.to(src_dtype) for t in self_tensors]
                # 使用 foreach_copy_ 方法进行复制操作，预期使用快速路径并在 CUDA 上执行
                out = foreach_copy_(
                    (self_tensors, src_tensors), is_cuda=True, expect_fastpath=True
                )
                # 断言复制结果 out 应与预期的列表一致，其中每个 Tensor 使用 torch.empty_like(t).copy_(s) 方法进行复制
                self.assertEqual(
                    out,
                    [
                        torch.empty_like(t).copy_(s)
                        for t, s in zip(self_tensors, src_tensors)
                    ],
                )

    # 测试反向模式和正向模式的自动微分，如果支持的话
    @onlyCUDA
    @ops(
        foreach_unary_op_db
        + foreach_binary_op_db
        + foreach_pointwise_op_db
        + foreach_reduce_op_db
        + foreach_other_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=(torch.float64, torch.complex128),
    )
    # 参数化测试，设置 inplace 参数为 False 和 True，根据参数值生成测试名称
    @parametrize(
        "inplace", (False, True), name_fn=lambda x: "inplace" if x else "outplace"
    )
# TODO(crcrpar): Hide this inside torch/testing/_internal.
# 将此函数隐藏在torch/testing/_internal中。
# would end up adding another layer to `foreach_inputs_sample_func.__call__`
# 最终将另一层添加到`foreach_inputs_sample_func.__call__`中，
# so that we can use this function as something like the first argument of `filter` function.
# 以便我们可以将此函数用作`filter`函数的第一个参数。
# Even after moving this function to testing, I personally think it'd be better to check the error message.
# 即使将此函数移到测试中，我个人认为最好检查错误消息。
def check_autodiff_sample(op, sample, dtype, is_inplace):
    # Check conditions for specific operations and return corresponding error messages.
    # 检查特定操作的条件，并返回相应的错误消息。
    if op.name == "_foreach_abs" and is_inplace and dtype == torch.complex128:
        return False, "In-place abs is not supported for complex tensors."
    if op.name == "_foreach_sub" and (
        (
            isinstance(sample.args[0], list)
            and any(isinstance(a, bool) for a in sample.args[0])
        )
        or isinstance(sample.args[0], bool)
    ):
        return False, _BOOL_SUB_ERR_MSG
    if op.name == "_foreach_norm" and (not is_inplace):
        return (
            False,
            "Trying to set a forward gradient that has a different size than that of the original Tensor, "
            "this is not supported. Tensor is of size [] while the given forward gradient is of size [1, 1].",
        )
    rhs_arg_has_complex_number = sample.args and (
        (
            isinstance(sample.args[0], list)
            and any(isinstance(a, complex) for a in sample.args[0])
        )
        or (isinstance(sample.args[0], complex))
    )
    if rhs_arg_has_complex_number and dtype == torch.float64:
        if op.name in (
            "_foreach_clamp_max",
            "_foreach_clamp_min",
            "_foreach_maximum",
            "_foreach_minimum",
        ):
            return False, "clamp is not supported for complex types"
        if not is_inplace:
            return False, ""
        else:
            if op.name == "_foreach_pow":
                return False, "Found dtype Double but expected ComplexDouble"
            if op.name in (
                "_foreach_add",
                "_foreach_sub",
                "_foreach_mul",
                "_foreach_div",
            ):
                return (
                    False,
                    "result type ComplexDouble can't be cast to the desired output type Double",
                )
    # Default case where no error is found
    return True, ""


# Instantiate device type tests for TestForeach class in the current global scope.
# 在当前的全局范围内为TestForeach类实例化设备类型测试。
instantiate_device_type_tests(TestForeach, globals())

# Entry point for running tests if this script is executed directly.
# 如果直接执行此脚本，则作为运行测试的入口点。
if __name__ == "__main__":
    # Run all defined tests.
    # 运行所有定义的测试。
    run_tests()
```