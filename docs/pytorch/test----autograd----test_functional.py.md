# `.\pytorch\test\autograd\test_functional.py`

```py
# Owner(s): ["module: autograd"]

# 引入所需的模块和类
import types  # 导入types模块，用于创建命名空间
import unittest  # 导入unittest模块，用于编写和运行单元测试
import warnings  # 导入warnings模块，用于管理警告信息

import torch  # 导入PyTorch深度学习框架
import torch.autograd.functional as autogradF  # 导入PyTorch自动求导功能模块

from torch.testing._internal.common_cuda import TEST_CUDA  # 导入测试CUDA相关的模块
from torch.testing._internal.common_utils import (  # 导入通用的测试工具函数和类
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.logging_tensor import LoggingTensor  # 导入用于记录张量操作的LoggingTensor类

# 用于自动求导测试中的张量构造函数的参数化工具
#
# TODO: 可能会移动到其他地方以便其他测试也能使用
#
# NB: 并非所有工厂函数都包含在内。完整的列表可以在此找到：
#     https://pytorch.org/cppdocs/notes/tensor_creation.html
base_ctors_dict = {
    "ones": torch.ones,   # 创建全为1的张量的工厂函数
    "zeros": torch.zeros,  # 创建全为0的张量的工厂函数
    "randn": torch.randn,  # 创建服从标准正态分布的张量的工厂函数
    "rand": torch.rand,    # 创建均匀分布的随机张量的工厂函数
    "tensor": torch.tensor,  # 根据给定数据创建张量的工厂函数
}
base_ctors = types.SimpleNamespace(**base_ctors_dict)  # 创建命名空间，包含上述张量工厂函数作为属性

# 将张量构造函数包装为带有记录功能的张量构造函数
def wrap_with_logging_tensor(ctor):
    def wrapper(*args, **kwargs):
        requires_grad = kwargs.pop("requires_grad", False)
        return LoggingTensor(ctor(*args, **kwargs), requires_grad=requires_grad)
    return wrapper

# 使用记录功能包装所有基本张量构造函数
logging_tensor_ctors_dict = {
    k: wrap_with_logging_tensor(ctor) for (k, ctor) in base_ctors_dict.items()
}
logging_tensor_ctors = types.SimpleNamespace(**logging_tensor_ctors_dict)  # 创建命名空间，包含记录功能张量构造函数作为属性

# 参数化测试：包括基本张量构造函数和记录功能张量构造函数
base_and_logging_tensor = parametrize(
    "ctors",
    [
        subtest(base_ctors, name="base_tensor"),  # 基本张量构造函数的子测试
        subtest(logging_tensor_ctors, name="logging_tensor"),  # 记录功能张量构造函数的子测试
    ],
)

# 参数化测试：包括基本张量构造函数和记录功能张量构造函数（预期失败）
FIXME_base_and_xfail_logging_tensor = parametrize(
    "ctors",
    [
        subtest(base_ctors, name="base_tensor"),  # 基本张量构造函数的子测试
        subtest(
            logging_tensor_ctors,
            name="logging_tensor",
            decorators=[unittest.expectedFailure],  # 预期此测试失败
        ),
    ],
)

# 参数化测试：包括向量化测试和记录功能张量构造函数（预期失败）
FIXME_xfail_vectorized_logging_tensor = parametrize(
    "vectorize,ctors",
    [
        subtest((True, base_ctors), name="vectorized_base_tensor"),  # 向量化的基本张量构造函数的子测试
        subtest((False, base_ctors), name="base_tensor"),  # 基本张量构造函数的子测试
        subtest(
            (True, logging_tensor_ctors),
            name="vectorized_logging_tensor",
            decorators=[unittest.expectedFailure],  # 预期此测试失败
        ),
        subtest((False, logging_tensor_ctors), name="logging_tensor"),  # 记录功能张量构造函数的子测试
    ],
)

# 参数化测试：包括向量化测试和记录功能张量构造函数
vectorized_logging_tensor = parametrize(
    "vectorize,ctors",
    [
        subtest((True, base_ctors), name="vectorized_base_tensor"),  # 向量化的基本张量构造函数的子测试
        subtest((False, base_ctors), name="base_tensor"),  # 基本张量构造函数的子测试
        subtest((True, logging_tensor_ctors), name="vectorized_logging_tensor"),  # 向量化的记录功能张量构造函数的子测试
        subtest((False, logging_tensor_ctors), name="logging_tensor"),  # 记录功能张量构造函数的子测试
    ],
)

# 自动求导功能测试类，继承于unittest的TestCase类
class TestAutogradFunctional(TestCase):
    # 确保 `res` 和 `base` 具有相同的数据结构和大小
    def _assert_same_struct(self, res, base):
        # 如果 `base` 是一个 Tensor
        if isinstance(base, torch.Tensor):
            # 确保 `res` 也是一个 Tensor
            self.assertTrue(isinstance(res, torch.Tensor))
            # 检查它们的大小是否相同
            self.assertEqual(base.size(), res.size())
        # 如果 `base` 是一个元组
        elif isinstance(base, tuple):
            # 确保 `res` 也是一个元组
            self.assertTrue(isinstance(res, tuple))
            # 检查元组的长度是否相同
            self.assertEqual(len(base), len(res))
            # 逐个检查元组中每对对应的元素
            for el_base, el_res in zip(base, res):
                # 每个元素都应该是 Tensor
                self.assertTrue(isinstance(el_base, torch.Tensor))
                self.assertTrue(isinstance(el_res, torch.Tensor))
                # 检查每对元素的大小是否相同
                self.assertEqual(el_base.size(), el_res.size())
        else:
            # 如果 `base` 类型不是预期的 Tensor 或元组，则抛出异常
            raise RuntimeError(
                "The base given to `_assert_same_struct` doesn't have"
                " the right structure."
            )
    def _assert_interleaved_struct(self, res, base1, base2):
        # base1 and base2 can be Tensors or tuples of Tensors.
        # If they are tuples, res should be a tuple as well.
        # The indexing works as follows for base1, base2 being
        # - tuple, tuple: res[i][j][k][l] = (base1[i][k], base2[j][l])
        # - tuple, Tensor: res[i][k][l] = (base1[i][k], base2[l])
        # - Tensor, tuple: res[i][j][l] = (base1[i], base2[j][l])
        # - Tensor, Tensor: res[k][l] = (base1[k], base2[l])

        # 根据不同的输入类型进行断言检查，确保输出 res 的结构满足预期
        if isinstance(base1, torch.Tensor) and isinstance(base2, torch.Tensor):
            # 如果 base1 和 base2 都是 Tensor 类型
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(res.size(), base1.size() + base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, torch.Tensor):
            # 如果 base1 是元组，base2 是 Tensor 类型
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            # 遍历检查每个元素的类型和尺寸
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base1, torch.Tensor))
                self.assertEqual(el_res.size(), el_base1.size() + base2.size())
        elif isinstance(base1, torch.Tensor) and isinstance(base2, tuple):
            # 如果 base1 是 Tensor 类型，base2 是元组
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base2))
            # 遍历检查每个元素的类型和尺寸
            for el_res, el_base2 in zip(res, base2):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base2, torch.Tensor))
                self.assertEqual(el_res.size(), base1.size() + el_base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, tuple):
            # 如果 base1 和 base2 都是元组
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            # 遍历检查每个元素的类型和尺寸
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, tuple))
                self.assertEqual(len(el_res), len(base2))
                # 遍历检查每个元素的类型和尺寸
                for el_el_res, el_base2 in zip(el_res, base2):
                    self.assertTrue(isinstance(el_el_res, torch.Tensor))
                    self.assertTrue(isinstance(el_base2, torch.Tensor))
                    self.assertEqual(
                        el_el_res.size(), el_base1.size() + el_base2.size()
                    )
        else:
            # 如果 base1 和 base2 的类型不匹配，抛出异常
            raise RuntimeError(
                "The bases given to `_assert_interleaved_struct` don't have"
                " the right structure."
            )

    @base_and_logging_tensor
    # 在给定的构造函数（ctors）上执行错误检查的测试方法
    def test_vjp_err_check(self, ctors):
        # 定义一个简单的函数 foo，返回输入张量的一个窄视图的三倍
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        # 定义一个复杂一点的函数 bar，返回输入张量的一个窄视图的三倍以及字符串 "bar"
        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        # 生成一个随机输入张量
        inp = ctors.rand(4)
        # 创建一个全为1的张量 v
        v = ctors.ones(3)

        # 断言调用 autogradF.vjp(foo, (inp, 2), v) 时抛出 TypeError 异常，指定异常消息为 "The inputs given to vjp must be either a Tensor"
        with self.assertRaisesRegex(
            TypeError, "The inputs given to vjp must be either a Tensor"
        ):
            res = autogradF.vjp(foo, (inp, 2), v)

        # 断言调用 autogradF.vjp(bar, inp, v) 时抛出 TypeError 异常，指定异常消息为 "The outputs of the user-provided function given to vjp must"
        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to vjp must"
        ):
            res = autogradF.vjp(bar, inp, v)

        # 断言调用 autogradF.vjp(foo, inp) 时抛出 RuntimeError 异常，指定异常消息为 "The vector v can only be None if the user-provided function returns"
        with self.assertRaisesRegex(
            RuntimeError,
            "The vector v can only be None if the user-provided function returns",
        ):
            res = autogradF.vjp(foo, inp)

        # 断言调用 autogradF.vjp(foo, inp, (torch.ones_like(inp), torch.ones_like(inp))) 时抛出 RuntimeError 异常，指定异常消息为 "The given v should contain a single Tensor."
        with self.assertRaisesRegex(
            RuntimeError, "The given v should contain a single Tensor."
        ):
            res = autogradF.vjp(foo, inp, (torch.ones_like(inp), torch.ones_like(inp)))

        # 断言调用 autogradF.vjp(foo, inp, v[:2]) 时抛出 RuntimeError 异常，指定异常消息为 "v has invalid size: should be torch.Size"
        with self.assertRaisesRegex(
            RuntimeError, "v has invalid size: should be torch.Size"
        ):
            res = autogradF.vjp(foo, inp, v[:2])

        # 调用 autogradF.vjp(foo, inp, v) 并获取其返回结果的第二个元素，断言其结构与输入 inp 相同
        res = autogradF.vjp(foo, inp, v)[1]
        self._assert_same_struct(res, inp)

    @base_and_logging_tensor
    # 在严格模式下执行错误检查的测试方法
    def test_vjp_err_check_strict(self, ctors):
        # 定义一个函数 foo，返回输入张量的一个剥离（detach）版本
        def foo(a):
            return a.detach()

        # 定义一个函数 bar，创建一个非叶子张量，其需要梯度，但与输入无关
        def bar(a):
            return a.long().float().requires_grad_().clone()

        # 生成一个随机输入张量
        inp = ctors.rand(4)
        # 创建一个随机张量 v
        v = ctors.rand(4)

        # 断言调用 autogradF.vjp(foo, inp, v, strict=True) 时抛出 RuntimeError 异常，指定异常消息为 "Output 0 of the user-provided function does not require gradients."
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.vjp(foo, inp, v, strict=True)

        # 调用 autogradF.vjp(foo, inp, v, strict=False) 并断言其返回结果的第二个元素结构与输入 inp 相同
        res = autogradF.vjp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 断言调用 autogradF.vjp(bar, inp, v, strict=True) 时抛出 RuntimeError 异常，指定异常消息为 "The output of the user-provided function is independent of input 0"
        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.vjp(bar, inp, v, strict=True)

        # 调用 autogradF.vjp(bar, inp, v, strict=False) 并断言其返回结果的第二个元素结构与输入 inp 相同
        res = autogradF.vjp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 重新定义函数 foo，返回输入张量的一个克隆版本
        def foo(a):
            return a.clone()

        # 将输入张量设置为需要梯度
        inp.requires_grad_()

        # 断言调用 autogradF.vjp(foo, inp, v, create_graph=True, strict=True) 时抛出 RuntimeError 异常，指定异常消息为 "jacobian of the user-provided function is independent of input 0."
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.vjp(foo, inp, v, create_graph=True, strict=True)

        # 调用 autogradF.vjp(foo, inp, v, create_graph=True, strict=False) 并断言其返回结果的第二个元素结构与输入 inp 相同，且等于输入的张量 v
        res = autogradF.vjp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)
    def test_vjp_no_grad(self, ctors):
        # 定义一个函数，用于对输入张量沿着第一维求和
        def reducer(x):
            return x.sum(dim=1)

        # 生成一个随机的 4x4 张量作为输入
        inputs = ctors.rand(4, 4)
        # 生成一个全为1的长度为4的张量
        v = ctors.ones(4)
        # 使用 torch.no_grad() 上下文管理器，禁止梯度计算
        with torch.no_grad():
            # 调用自动求导函数 autogradF.vjp，计算 reducer 函数关于 inputs 和 v 的向量-雅可比积
            res = autogradF.vjp(reducer, inputs, v)
        # 断言结果的梯度函数为空
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        # 断言结果中的第二个元素不等于全零张量
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        # 将 inputs 和 v 设置为需要计算梯度
        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            # 再次调用 autogradF.vjp，此时设置 create_graph=True，以便创建计算图
            res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        # 断言结果的梯度函数不为空
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        # 断言结果中的第二个元素不等于全零张量
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_vjp_output(self, ctors):
        # 定义一个函数，用于对输入张量沿着第一维求和
        def reducer(x):
            return x.sum(dim=1)

        # 生成一个随机的 4x4 张量作为输入
        inputs = ctors.rand(4, 4)
        # 生成一个全为1的长度为4的张量
        v = ctors.ones(4)
        # 调用 autogradF.vjp 计算 reducer 函数关于 inputs 和 v 的向量-雅可比积
        res = autogradF.vjp(reducer, inputs, v)
        # 断言结果的梯度函数为空
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        # 定义一个函数，用于对两个输入张量进行加权和计算
        def adder(x, y):
            return 2 * x + 3 * y

        # 生成两个随机的长度为2的张量作为输入
        inputs = (ctors.rand(2), ctors.rand(2))
        # 生成一个全为1的长度为2的张量
        v = ctors.ones(2)
        # 调用 autogradF.vjp 计算 adder 函数关于 inputs 和 v 的向量-雅可比积
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        # 断言输出结果与 vjp_val 的结构相同
        self._assert_same_struct(vjp_val, inputs)
        # 断言输出结果的梯度函数为空
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

        # 定义一个函数，返回两个加权和计算的结果
        def adder(x, y):
            return 2 * x + 3 * y, x + y

        # 生成两个随机的长度为2的张量作为输入
        inputs = (ctors.rand(2), ctors.rand(2))
        # 生成两个张量，分别为 [1.0, 0.0] 和 [1.0, 0.0]
        v = (ctors.tensor([1.0, 0.0]), ctors.tensor([1.0, 0.0]))
        # 调用 autogradF.vjp 计算 adder 函数关于 inputs 和 v 的向量-雅可比积
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        # 断言输出结果与 vjp_val 的结构相同
        self._assert_same_struct(vjp_val, inputs)
        # 断言输出结果中的第一个元素的梯度函数为空
        self.assertIsNone(out[0].grad_fn)
        # 断言输出结果中的第二个元素的梯度函数为空
        self.assertIsNone(out[1].grad_fn)
        # 断言 vjp_val 中的第一个元素的梯度函数为空
        self.assertIsNone(vjp_val[0].grad_fn)
        # 断言 vjp_val 中的第二个元素的梯度函数为空
        self.assertIsNone(vjp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_vjp_scalar(self, ctors):
        # 定义一个函数，用于对输入张量求和
        def reducer(x):
            return x.sum()

        # 生成一个随机的 4x4 张量作为输入
        inputs = ctors.rand(4, 4)
        # 生成一个标量值为1的张量
        v = ctors.ones([])
        # 调用 autogradF.vjp 计算 reducer 函数关于 inputs 的梯度和 v 的向量-雅可比积
        res = autogradF.vjp(reducer, inputs, v)
        # 断言输出结果的第一个元素结构与 v 相同
        self._assert_same_struct(res[0], v)
        # 断言输出结果的第二个元素结构与 inputs 相同

        # 再次调用 autogradF.vjp，但没有指定 v，此时 v 默认为标量1
        res = autogradF.vjp(reducer, inputs)
        # 断言输出结果的第一个元素结构与 v 相同
        self._assert_same_struct(res[0], v)
        # 断言输出结果的第二个元素结构与 inputs 相同

        # 定义一个函数，用于将输入张量扩展为长度为4的张量
        def expander(x):
            return x.unsqueeze(0).repeat(4)

        # 生成一个随机的标量张量作为输入
        inputs = ctors.rand([])
        # 生成一个全为1的长度为4的张量
        v = ctors.ones(4)
        # 调用 autogradF.vjp 计算 expander 函数关于 inputs 和 v 的向量-雅可比积
        res = autogradF.vjp(expander, inputs, v)
        # 断言输出结果的第一个元素结构与 v 相同
        self._assert_same_struct(res[0], v)
        # 断言输出结果的第二个元素结构与 inputs 相同
    # 定义测试方法，用于验证反向传播中的向量雅可比积（VJP）的创建图功能
    def test_vjp_create_graph(self, ctors):
        # 定义一个简化器函数，对输入张量在第一个维度上求和
        def reducer(x):
            return x.sum(dim=1)

        # 创建随机输入张量并设置数据类型为双精度浮点型
        inputs = ctors.rand(2, 2, dtype=torch.double)
        # 创建值为1的张量作为v向量，并设置数据类型为双精度浮点型
        v = ctors.ones(2, dtype=torch.double)

        # 将输入张量和v向量设置为需要计算梯度
        inputs.requires_grad_()
        v.requires_grad_()
        # 调用自动求导函数autogradF.vjp，计算reducer函数关于inputs和v的VJP，并创建计算图
        res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        # 断言res的第一个元素与inputs具有相同的结构
        self._assert_same_struct(res[1], inputs)
        # 断言res的第一个元素具有梯度函数
        self.assertIsNotNone(res[0].grad_fn)
        # 断言res的第二个元素具有梯度函数
        self.assertIsNotNone(res[1].grad_fn)

        # 对autogradF.vjp函数的输入进行梯度检查
        gradcheck(
            lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True),
            (inputs, v),
        )
        # 对autogradF.vjp函数的输入进行二阶梯度检查
        gradgradcheck(
            lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True),
            (inputs, v),
        )

        # 定义一个adder函数，接受两个参数x和y，并返回2*x + 3*y和x*y
        def adder(x, y):
            return 2 * x + 3 * y, x * y

        # 创建随机输入张量inputs，每个张量都需要计算梯度，数据类型为双精度浮点型
        inputs = (
            ctors.rand(2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, dtype=torch.double, requires_grad=True),
        )
        # 创建两个张量v，分别为[1.0, 0.0]和[1.0, 0.0]，都需要计算梯度，数据类型为双精度浮点型
        v = (
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
        )

        # 对adder函数的输入进行梯度检查
        gradcheck(
            lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )
        # 对adder函数的输入进行二阶梯度检查
        gradgradcheck(
            lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[
                1
            ],
            inputs + v,
        )

        # 定义一个foo函数，接受任意数量的参数args
        def foo(*args):
            # 将args的前两个参数作为x和y
            x, y = args[:2]
            # 将args的剩余参数作为v
            v = args[2:]

            # 对x取余弦函数
            x = x.cos()
            # 调用autogradF.vjp函数计算adder函数关于(x, y)和v的VJP，并创建计算图
            val, grad = autogradF.vjp(adder, (x, y), v, create_graph=True)

            # 返回值为val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()
            return (
                val[0].exp()
                + val[1].exp()
                + grad[0].exp()
                + grad[1].exp()
                + x.exp()
                + y.exp()
            )

        # 对foo函数的输入进行梯度检查
        gradcheck(foo, inputs + v)
        # 对foo函数的输入进行二阶梯度检查
        gradgradcheck(foo, inputs + v)
    # 定义一个测试函数，用于检查 jvp 函数的错误处理机制，接受 ctors 参数作为构造函数
    def test_jvp_err_check(self, ctors):
        # 定义内部函数 foo，对输入 a 进行窄化操作并返回结果的三倍
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        # 定义内部函数 bar，对输入 a 进行窄化操作并返回结果的三倍，并返回字符串 "bar"
        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        # 生成随机输入 inp 和 v
        inp = ctors.rand(4)
        v = ctors.rand(4)

        # 测试使用 jvp 函数调用 foo 函数，期望抛出 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "The inputs given to jvp must be either a Tensor"
        ):
            res = autogradF.jvp(foo, (inp, 2), v)

        # 测试使用 jvp 函数调用 bar 函数，期望抛出 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to jvp must"
        ):
            res = autogradF.jvp(bar, inp, v)

        # 测试使用 jvp 函数调用 foo 函数，但没有传入 v，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "The vector v can only be None if the input to the user-provided function",
        ):
            res = autogradF.jvp(foo, inp)

        # 测试使用 jvp 函数调用 foo 函数，但传入的 v 中包含多个 Tensor，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "The given v should contain a single Tensor."
        ):
            res = autogradF.jvp(foo, inp, (v, v))

        # 测试使用 jvp 函数调用 foo 函数，但传入的 v 大小不合法，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "v has invalid size: should be torch.Size"
        ):
            res = autogradF.jvp(foo, inp, v[:2])

        # 调用 jvp 函数计算 foo 函数对输入 inp 和 v 的结果，并取结果的第二个元素
        res = autogradF.jvp(foo, inp, v)[1]
        # 断言 res 与 foo(inp) 具有相同的结构
        self._assert_same_struct(res, foo(inp))

    # 用装饰器 base_and_logging_tensor 包装的测试函数
    @base_and_logging_tensor
    # 定义严格模式下的 jvp 错误检查函数，接受 ctors 参数作为构造函数
    def test_jvp_err_check_strict(self, ctors):
        # 定义内部函数 foo，对输入 a 进行 detach 操作并返回结果
        def foo(a):
            return a.detach()

        # 定义内部函数 bar，创建一个非叶子张量，要求梯度但与输入无关
        def bar(a):
            return a.long().float().requires_grad_().clone()

        # 生成随机输入 inp 和 v
        inp = ctors.rand(4)
        v = ctors.rand(4)

        # 测试使用 jvp 函数调用 foo 函数，严格模式下期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.jvp(foo, inp, v, strict=True)

        # 调用 jvp 函数计算 foo 函数对输入 inp 和 v 的结果，非严格模式
        res = autogradF.jvp(foo, inp, v, strict=False)
        # 断言 res 的第二个元素与 res 的第一个元素具有相同的结构
        self._assert_same_struct(res[1], res[0])
        # 断言 res 的第二个元素的绝对值之和为 0.0
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 测试使用 jvp 函数调用 bar 函数，严格模式下期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.jvp(bar, inp, v, strict=True)

        # 调用 jvp 函数计算 bar 函数对输入 inp 和 v 的结果，非严格模式
        res = autogradF.jvp(bar, inp, v, strict=False)
        # 断言 res 的第二个元素与 res 的第一个元素具有相同的结构
        self._assert_same_struct(res[1], res[0])
        # 断言 res 的第二个元素的绝对值之和为 0.0
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 定义内部函数 foo，复制输入 a 并返回结果
        def foo(a):
            return a.clone()

        # 将输入 inp 设置为需要梯度
        inp.requires_grad_()
        # 测试使用 jvp 函数调用 foo 函数，严格模式下期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jvp(foo, inp, v, create_graph=True, strict=True)

        # 调用 jvp 函数计算 foo 函数对输入 inp 和 v 的结果，非严格模式，并创建计算图
        res = autogradF.jvp(foo, inp, v, create_graph=True, strict=False)
        # 断言 res 的第二个元素与 inp 具有相同的结构
        self._assert_same_struct(res[1], inp)
        # 断言 res 的第二个元素与 v 相等
        self.assertEqual(res[1], v)
    # 测试无梯度的 JVP（雅可比向量积），传入参数为 ctors
    def test_jvp_no_grad(self, ctors):
        # 定义一个简单的 reducer 函数，对输入的张量沿着第一维求和
        def reducer(x):
            return x.sum(dim=1)

        # 生成一个随机的输入张量
        inputs = ctors.rand(4, 4)
        # 创建一个全为1的张量作为向量 v
        v = ctors.ones(4, 4)
        # 使用 torch.no_grad() 上下文管理器，确保计算过程中不会记录梯度信息
        with torch.no_grad():
            # 计算 reducer 函数的 JVP，即 reducer 在 inputs 上对 v 的雅可比向量积
            res = autogradF.jvp(reducer, inputs, v)
        # 断言 res 的第一个输出的梯度函数为空，即无梯度
        self.assertIsNone(res[0].grad_fn)
        # 断言 res 的第二个输出的梯度函数为空，即无梯度
        self.assertIsNone(res[1].grad_fn)
        # 断言 res 的第二个输出不等于一个全为0的张量，即验证其结果与预期不同

        # 将 inputs 和 v 设置为需要计算梯度的张量
        inputs.requires_grad_()
        v.requires_grad_()
        # 再次使用 torch.no_grad() 上下文管理器，创建计算图
        with torch.no_grad():
            # 计算 reducer 函数的 JVP，此时 create_graph=True，即创建计算图
            res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        # 断言 res 的第一个输出的梯度函数不为空，即存在梯度信息
        self.assertIsNotNone(res[0].grad_fn)
        # 断言 res 的第二个输出的梯度函数不为空，即存在梯度信息
        self.assertIsNotNone(res[1].grad_fn)
        # 断言 res 的第二个输出不等于一个全为0的张量，即验证其结果与预期不同

    @base_and_logging_tensor
    # 测试 JVP 的输出结果结构
    def test_jvp_output(self, ctors):
        # 定义一个简单的 reducer 函数，对输入的张量沿着第一维求和
        def reducer(x):
            return x.sum(dim=1)

        # 生成一个随机的输入张量
        inputs = ctors.rand(4, 4)
        # 创建一个全为1的张量作为向量 v
        v = ctors.ones(4, 4)
        # 计算 reducer 函数的 JVP
        res = autogradF.jvp(reducer, inputs, v)
        # 断言 res 的第一个输出的梯度函数为空，即无梯度
        self.assertIsNone(res[0].grad_fn)
        # 断言 res 的第二个输出的梯度函数为空，即无梯度

        # 定义一个简单的 adder 函数，对两个输入张量进行加权和
        def adder(x, y):
            return 2 * x + 3 * y

        # 生成两个随机输入张量和全为1的向量作为输入
        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.ones(2), ctors.ones(2))
        # 计算 adder 函数的 JVP
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        # 断言 jvp_val 和 out 结构相同
        self._assert_same_struct(jvp_val, out)
        # 断言 out 的梯度函数为空，即无梯度
        self.assertIsNone(out.grad_fn)
        # 断言 jvp_val 的两个分量的梯度函数为空，即无梯度

        # 重新定义 adder 函数，返回两个结果
        def adder(x, y):
            return 2 * x + 3 * y, x + y

        # 生成两个随机输入张量和全为1的向量作为输入
        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.tensor([1.0, 0.0]), ctors.tensor([1.0, 0.0]))
        # 计算 adder 函数的 JVP
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        # 断言 jvp_val 和 out 结构相同
        self._assert_same_struct(jvp_val, out)
        # 断言 out 的第一个输出的梯度函数为空，即无梯度
        self.assertIsNone(out[0].grad_fn)
        # 断言 out 的第二个输出的梯度函数为空，即无梯度
        self.assertIsNone(out[1].grad_fn)
        # 断言 jvp_val 的两个分量的梯度函数为空，即无梯度

    @base_and_logging_tensor
    # 测试 JVP 处理标量的情况
    def test_jvp_scalar(self, ctors):
        # 定义一个简单的 reducer 函数，对输入的张量求和
        def reducer(x):
            return x.sum()

        # 生成一个随机的输入张量
        inputs = ctors.rand(4, 4)
        # 创建一个全为1的张量作为向量 v
        v = ctors.ones(4, 4)
        # 计算 reducer 函数的 JVP
        res = autogradF.jvp(reducer, inputs, v)
        # 断言 res 的第一个输出为一个空的标量张量
        self._assert_same_struct(res[0], ctors.zeros([]))
        # 断言 res 的第二个输出结构与第一个输出相同

        # 定义一个简单的 expander 函数，将标量张量扩展成一个向量
        def expander(x):
            return x.unsqueeze(0).repeat(4)

        # 生成一个随机的标量输入张量
        inputs = ctors.rand([])
        # 创建一个全为1的标量张量作为向量 v
        v = ctors.ones([])
        # 计算 expander 函数的 JVP
        res = autogradF.jvp(expander, inputs, v)
        # 断言 res 的第一个输出为一个全为0的长度为4的向量
        self._assert_same_struct(res[0], ctors.zeros(4))
        # 断言 res 的第二个输出结构与第一个输出相同

        # 再次计算 expander 函数的 JVP，不传入向量 v
        res = autogradF.jvp(expander, inputs)
        # 断言 res 的第一个输出为一个全为0的长度为4的向量
        self._assert_same_struct(res[0], ctors.zeros(4))
        # 断言 res 的第二个输出结构与第一个输出相同
    # 定义一个测试方法，用于测试 autogradF.jvp 方法的创建图功能
    def test_jvp_create_graph(self, ctors):
        # 定义一个简单的函数，对输入张量按行求和
        def reducer(x):
            return x.sum(dim=1)

        # 创建随机张量作为输入和全1张量作为向量
        inputs = ctors.rand(2, 2, dtype=torch.double)
        v = ctors.ones(2, 2, dtype=torch.double)

        # 将输入张量和全1张量设置为需要梯度计算
        inputs.requires_grad_()
        v.requires_grad_()

        # 使用 autogradF.jvp 计算 reducer 函数的 JVP（雅可比向量积），并创建计算图
        res = autogradF.jvp(reducer, inputs, v, create_graph=True)

        # 断言结果的结构与输入相同
        self._assert_same_struct(res[1], res[0])

        # 断言结果的梯度函数不为空
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        # 对 autogradF.jvp 函数进行梯度检查
        gradcheck(
            lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True),
            (inputs, v),
        )
        
        # 对 autogradF.jvp 函数进行二阶梯度检查
        gradgradcheck(
            lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True),
            (inputs, v),
        )

        # 定义一个简单的函数，对两个输入张量进行加权求和，并返回两个结果的乘积
        def adder(x, y):
            return 2 * x + 3 * y, x * y

        # 创建随机张量和全1张量作为输入，并设置需要梯度计算
        inputs = (
            ctors.rand(2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, dtype=torch.double, requires_grad=True),
        )
        v = (
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
            ctors.tensor([1.0, 0.0], dtype=torch.double, requires_grad=True),
        )

        # 对 autogradF.jvp 函数进行梯度检查
        gradcheck(
            lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        # 对 autogradF.jvp 函数进行二阶梯度检查
        gradgradcheck(
            lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        # 定义一个函数，接受任意数量的输入，对前两个输入张量分别应用余弦函数，然后使用 autogradF.jvp 计算 adder 函数的 JVP
        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            # 对第一个输入张量应用余弦函数
            x = x.cos()
            
            # 使用 autogradF.jvp 计算 adder 函数的 JVP，并创建计算图
            val, grad = autogradF.jvp(adder, (x, y), v, create_graph=True)

            # 返回多个项的指数计算结果
            return (
                val[0].exp()
                + val[1].exp()
                + grad[0].exp()
                + grad[1].exp()
                + x.exp()
                + y.exp()
            )

        # 对 foo 函数进行梯度检查
        gradcheck(foo, inputs + v)
        
        # 对 foo 函数进行二阶梯度检查
        gradgradcheck(foo, inputs + v)

    # 测试方法，用于测试 autogradF._construct_standard_basis_for 方法
    def _test_construct_standard_basis_for(self, inputs):
        # 计算每个输入张量的元素数量
        numels = tuple(tensor.numel() for tensor in inputs)
        
        # 使用 autogradF._construct_standard_basis_for 方法构建标准基向量
        results = autogradF._construct_standard_basis_for(inputs, numels)
        
        # 断言每个结果张量与对应输入张量的数据类型和设备相同
        for result, inp in zip(results, inputs):
            self.assertEqual(result.dtype, inp.dtype)
            self.assertEqual(result.device, inp.device)

        # 将结果张量拼接成一个大张量，转换为 CPU 上的浮点型张量
        results = torch.cat(
            [result.to(device="cpu", dtype=torch.float) for result in results], dim=1
        )
        
        # 创建预期的单位矩阵
        expected = torch.eye(results[0].shape[0], dtype=torch.float)

        # 断言结果张量等于预期的单位矩阵
        self.assertEqual(results, expected)

    # 装饰器函数，用于基础和日志张量
    @base_and_logging_tensor
    def test_construct_standard_basis_for(self, ctors):
        # 定义测试用例，每个元组表示一个测试输入
        test_cases = [
            (ctors.randn(2, 3),),
            (ctors.randn(1),),
            (ctors.randn([]),),
            (ctors.randn(1), ctors.randn([]), ctors.randn([])),
            (ctors.randn(2), ctors.randn(3), ctors.randn([])),
            (ctors.randn(2), ctors.randn([]), ctors.randn(3)),
            (ctors.randn(2, 3), ctors.randn(3), ctors.randn(3, 4, 2)),
            (ctors.randn(2, dtype=torch.float64), ctors.randn(3, dtype=torch.float32)),
        ]

        # 遍历测试用例并调用 _test_construct_standard_basis_for 进行测试
        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @base_and_logging_tensor
    def test_construct_standard_basis_for_cuda(self, ctors):
        # 定义 CUDA 环境下的测试用例
        test_cases = [
            (ctors.randn(2), ctors.randn(3, device="cuda")),
            (ctors.randn(3, device="cuda"), ctors.randn(2)),
        ]

        # 遍历测试用例并调用 _test_construct_standard_basis_for 进行 CUDA 测试
        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    def _test_vectorize_raises_no_warnings(self, api, ctors):
        # _test_vectorize_raises_no_warnings 测试函数，检查调用 api 是否会引发警告
        # 定义一个简单的函数 foo 用于测试
        def foo(a):
            return (a**2).sum()

        # 生成随机输入
        x = ctors.randn(3)
        # 使用 warnings.catch_warnings 捕获警告信息
        with warnings.catch_warnings(record=True) as wa:
            # 调用 api 函数，这里是 autogradF.jacobian 或 autogradF.hessian
            result = api(foo, x, vectorize=True)
        # 断言没有警告被触发
        self.assertEqual(len(wa), 0)

    @base_and_logging_tensor
    def test_jacobian_vectorize_raises_no_warnings(self, ctors):
        # 对 autogradF.jacobian 调用 _test_vectorize_raises_no_warnings 进行测试
        return self._test_vectorize_raises_no_warnings(autogradF.jacobian, ctors)

    @base_and_logging_tensor
    def test_hessian_vectorize_raises_no_warnings(self, ctors):
        # 对 autogradF.hessian 调用 _test_vectorize_raises_no_warnings 进行测试
        return self._test_vectorize_raises_no_warnings(autogradF.hessian, ctors)

    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    @base_and_logging_tensor
    # 使用装饰器 `base_and_logging_tensor`，该装饰器可能会修改测试函数的行为和输出

    def test_jacobian_err_check(self, vectorize, ctors):
        # 定义内部函数 foo，计算输入张量的部分切片乘以 3
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        # 定义内部函数 bar，返回输入张量的部分切片乘以 3 和字符串 "bar"
        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        # 生成一个随机输入张量 inp，长度为 4
        inp = ctors.rand(4)
        
        # 使用 assertRaisesRegex 断言捕获 TypeError 异常，检查 jacobian 函数的输入参数类型
        with self.assertRaisesRegex(
            TypeError, "The inputs given to jacobian must be either a Tensor"
        ):
            res = autogradF.jacobian(foo, (inp, 2), vectorize=vectorize)

        # 使用 assertRaisesRegex 断言捕获 TypeError 异常，检查 jacobian 函数的输出类型
        with self.assertRaisesRegex(
            TypeError,
            "The outputs of the user-provided function given to jacobian must",
        ):
            res = autogradF.jacobian(bar, inp, vectorize=vectorize)

        # 调用 jacobian 函数计算函数 foo 的雅可比矩阵，并进行结构化断言
        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(inp), inp)

        # 重新定义函数 foo，接受两个输入参数 a 和 b，返回 b 和 a 的部分切片乘以 3
        def foo(a, b):
            return b, 3 * a.narrow(0, 0, 3)

        # 生成两个随机输入张量 inp，长度分别为 4 和 5
        inp = (ctors.rand(4), ctors.rand(5))

        # 调用 jacobian 函数计算函数 foo 的雅可比矩阵，并进行结构化断言
        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(*inp), inp)

    @base_and_logging_tensor
    # 使用装饰器 `base_and_logging_tensor`，该装饰器可能会修改测试函数的行为和输出

    def test_jacobian_err_check_strict(self, ctors):
        # 定义函数 foo，返回输入张量的 detach 结果
        def foo(a):
            return a.detach()

        # 定义函数 bar，创建一个不是叶子节点的张量，该张量需要梯度，但与输入无关
        def bar(a):
            return a.long().float().requires_grad_().clone()

        # 生成一个随机输入张量 inp，长度为 4
        inp = ctors.rand(4)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查函数 foo 的输出是否需要梯度
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.jacobian(foo, inp, strict=True)

        # 调用 jacobian 函数计算函数 foo 的雅可比矩阵，并进行结构化断言
        res = autogradF.jacobian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.0)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查函数 bar 的输出是否与输入无关
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jacobian(bar, inp, strict=True)

        # 调用 jacobian 函数计算函数 bar 的雅可比矩阵，并进行结构化断言
        res = autogradF.jacobian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.0)

        # 重新定义函数 foo，返回输入张量的克隆
        def foo(a):
            return a.clone()

        # 将输入张量标记为需要梯度
        inp.requires_grad_()

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查函数 foo 的雅可比矩阵是否与输入无关
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function is independent of input 0.",
        ):
            res = autogradF.jacobian(foo, inp, create_graph=True, strict=True)

        # 调用 jacobian 函数计算函数 foo 的雅可比矩阵，并进行结构化断言
        res = autogradF.jacobian(foo, inp, create_graph=True, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res, torch.eye(4))

    @base_and_logging_tensor
    # 使用装饰器 `base_and_logging_tensor`，该装饰器可能会修改测试函数的行为和输出

    def test_jacobian_err_check_strict_vectorize(self, ctors):
        # 定义函数 foo，返回输入张量本身
        def foo(x):
            return x

        # 生成一个随机输入张量 inp，长度为 4
        inp = ctors.rand(4)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查 vectorize 和 strict 参数是否同时支持
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.jacobian(foo, inp, strict=True, vectorize=True)
    # 定义一个测试方法，用于验证在没有梯度跟踪的情况下计算雅可比矩阵
    def test_jacobian_no_grad(self, ctors):
        # 定义一个函数，计算输入张量每行的指数和
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        # 生成一个随机张量作为输入
        inputs = ctors.rand(4, 4)
        # 在没有梯度跟踪的上下文中计算 exp_reducer 函数的雅可比矩阵
        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs)
        # 断言结果的梯度函数不存在
        self.assertIsNone(res.grad_fn)
        # 断言结果不等于一个全零张量
        self.assertNotEqual(res, ctors.zeros(4, 4))

        # 在没有梯度跟踪的上下文中，再次计算 exp_reducer 函数的雅可比矩阵，并创建计算图
        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs, create_graph=True)
        # 断言结果的梯度函数存在
        self.assertIsNotNone(res.grad_fn)
        # 断言结果不等于一个全零张量
        self.assertNotEqual(res, ctors.zeros(4, 4))

    # 使用装饰器 vectorized_logging_tensor 定义一个测试方法，用于验证雅可比矩阵的输出
    def test_jacobian_output(self, vectorize, ctors):
        # 定义一个函数，计算输入张量每行的指数和
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        # 生成一个随机张量作为输入，并计算 exp_reducer 函数的雅可比矩阵
        inputs = ctors.rand(4, 4)
        res = autogradF.jacobian(exp_reducer, inputs, vectorize=vectorize)
        # 断言结果与 exp_reducer 在输入上的结构化输出相匹配
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        # 断言结果的梯度函数不存在
        self.assertIsNone(res.grad_fn)

        # 定义一个函数，返回输入张量的克隆
        def identity(x):
            return x.clone()

        # 生成一个随机张量作为输入，并计算 identity 函数的雅可比矩阵
        inputs = ctors.rand(4)
        res = autogradF.jacobian(identity, inputs, vectorize=vectorize)
        # 断言结果与 identity 在输入上的结构化输出相匹配
        self._assert_interleaved_struct(res, identity(inputs), inputs)
        # 断言结果的梯度函数不存在
        self.assertIsNone(res.grad_fn)
        # 断言结果等于单位矩阵
        self.assertEqual(res, torch.eye(4))

        # 定义一个函数，计算输入张量的每行加上第二个输入张量的指数和
        def add_exp_reducer(x, y):
            return (x + y.exp()).sum(dim=1)

        # 生成两个随机张量作为输入，并计算 add_exp_reducer 函数的雅可比矩阵
        inputs = (ctors.rand(4, 4), ctors.rand(4, 4))
        res = autogradF.jacobian(add_exp_reducer, inputs, vectorize=vectorize)
        # 断言结果与 add_exp_reducer 在输入上的结构化输出相匹配
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        # 断言结果的梯度函数不存在（分别对两个输入进行检查）
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

    # 使用装饰器 vectorized_logging_tensor 定义一个测试方法，用于验证标量函数的雅可比矩阵
    def test_jacobian_scalar(self, vectorize, ctors):
        # 定义一个函数，计算输入张量的和
        def reducer(x):
            return x.sum()

        # 生成一个随机张量作为输入，并计算 reducer 函数的雅可比矩阵
        inputs = ctors.rand(4, 4)
        res = autogradF.jacobian(reducer, inputs, vectorize=vectorize)
        # 断言结果与输入张量的结构相同
        self._assert_same_struct(res, inputs)

        # 定义一个函数，扩展输入张量并重复四次
        def expander(x):
            return x.unsqueeze(0).repeat(4)

        # 生成一个随机标量作为输入，并计算 expander 函数的雅可比矩阵
        inputs = ctors.rand([])
        res = autogradF.jacobian(expander, inputs, vectorize=vectorize)
        # 断言结果与一个全零张量的结构相同
        self._assert_same_struct(res, ctors.zeros(4))
    # 测试函数：验证雅可比矩阵的创建和计算的正确性
    def test_jacobian_create_graph(self, vectorize, ctors):
        # 定义指数求和缩减函数
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        # 创建随机输入张量，要求计算梯度
        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        # 计算 exp_reducer 的输入张量 inputs 的雅可比矩阵，并创建计算图
        res = autogradF.jacobian(
            exp_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        # 断言 res 结果的结构和预期的一致，并验证输入是否正确交错
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        # 确认 res 的梯度函数存在
        self.assertIsNotNone(res.grad_fn)

        # 对 exp_reducer 函数进行梯度检查
        gradcheck(
            lambda inp: autogradF.jacobian(
                exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        # 对 exp_reducer 函数进行二阶梯度检查
        gradgradcheck(
            lambda inp: autogradF.jacobian(
                exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        # 定义带有加法的指数求和缩减函数
        def add_exp_reducer(x, y):
            return (x + y).exp().sum(dim=1)

        # 创建两个随机输入张量，要求计算梯度
        inputs = (
            ctors.rand(4, 4, dtype=torch.double, requires_grad=True),
            ctors.rand(4, 4, dtype=torch.double, requires_grad=True),
        )
        # 计算 add_exp_reducer 的输入张量 inputs 的雅可比矩阵，并创建计算图
        res = autogradF.jacobian(
            add_exp_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        # 断言 res 结果的结构和预期的一致，并验证输入是否正确交错
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        # 确认 res 中每个输入的梯度函数存在
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        # 对 add_exp_reducer 函数进行梯度检查
        gradcheck(
            lambda *inp: autogradF.jacobian(
                add_exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        # 对 add_exp_reducer 函数进行二阶梯度检查
        gradgradcheck(
            lambda *inp: autogradF.jacobian(
                add_exp_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        # 定义复合函数 foo
        def foo(x, y):
            # 对 x 应用余弦函数
            x = x.cos()
            # 计算 add_exp_reducer 的输入 (x, y) 的值和雅可比矩阵，并创建计算图
            val, jac = autogradF.jacobian(
                add_exp_reducer, (x, y), create_graph=True, vectorize=vectorize
            )

            # 计算多个张量的指数求和，并将结果累加
            res = val[0].exp().sum() + val[1].exp().sum() + jac[0].exp().sum()
            res = res + jac[1].exp().sum() + x.exp().sum() + y.exp().sum()
            return res

        # 对 foo 函数进行梯度检查
        gradcheck(foo, inputs)
        # 对 foo 函数进行二阶梯度检查
        gradgradcheck(foo, inputs)

    # 辅助函数：检查矢量化正确性的雅可比矩阵
    def _check_jacobian_vectorize_correctness(self, f, inputs, test_forward_ad=True):
        # 计算非矢量化版本的 f 的雅可比矩阵
        expected = autogradF.jacobian(f, inputs, vectorize=False)
        # 计算使用矢量化策略计算的 f 的雅可比矩阵
        result_backward_mode = autogradF.jacobian(f, inputs, vectorize=True)
        # 断言矢量化后的计算结果与非矢量化版本一致
        self.assertEqual(result_backward_mode, expected)

        if test_forward_ad:
            # 如果测试前向自动求导策略
            # 计算使用前向自动求导策略和矢量化策略计算的 f 的雅可比矩阵
            result_forward_mode = autogradF.jacobian(
                f, inputs, strategy="forward-mode", vectorize=True
            )
            # 断言前向自动求导策略计算的结果与非矢量化版本一致
            self.assertEqual(result_forward_mode, expected)

    # 测试函数：简单验证矢量化正确性的雅可比矩阵
    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_simple(self, ctors):
        # 定义函数 f(x) = 3 * x**2
        def f(x):
            return 3 * x**2

        # 创建随机张量 x
        x = ctors.randn(2, 3, 5)
        # 使用辅助函数验证 f 的矢量化正确性
        self._check_jacobian_vectorize_correctness(f, x)
    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_multi_input(self, ctors):
        # 定义函数 f，接受两个输入 x 和 y，计算 x 的余弦乘以 x，再与 y 的正弦乘积的点积
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        # 生成一个大小为 (2, 3) 的随机张量 x
        x = ctors.randn(2, 3)
        # 生成一个大小为 (3, 5) 的随机张量 y
        y = ctors.randn(3, 5)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_multi_input_multi_output(self, ctors):
        # 定义函数 f，接受两个输入 x 和 y，返回三个输出：x 与 y 的平方的点积，x 与（x 每行求和乘以 y）的点积，以及 y 的总和
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        # 生成一个大小为 (5, 3) 的随机张量 x
        x = ctors.randn(5, 3)
        # 生成一个大小为 (3, 5) 的随机张量 y
        y = ctors.randn(3, 5)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_unrelated_outputs(self, ctors):
        # 定义函数 f，接受两个输入 x 和 y，返回四个输出：x, y, x, y
        def f(x, y):
            return x, y, x, y

        # 生成一个大小为 (2,) 的随机张量 x
        x = ctors.randn(2)
        # 生成一个大小为 (3,) 的随机张量 y
        y = ctors.randn(3)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_zero_dim(self, ctors):
        # zero-dim output
        # 定义函数 f，接受两个输入 x 和 y，返回三个输出：x 的和，y 的和，以及 x 与 y 的逐元素乘积
        def f(x, y):
            return x.sum(), y.sum(), x * y

        # 生成一个大小为 (3,) 的随机张量 x
        x = ctors.randn(3)
        # 生成一个大小为 (3,) 的随机张量 y
        y = ctors.randn(3)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(f, (x, y))

        # zero-dim input
        # 定义函数 g，接受一个输入 x，返回一个张量，其中包含 x 的三个副本
        def g(x):
            return torch.stack([x, x, x])

        # 生成一个标量张量 x
        x = ctors.randn([])
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 g 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(g, x)

        # Mixed zero-dim input / zero-dim output
        # 定义函数 h，接受两个输入 x 和 y，返回两个输出：y 的和，以及 x 与 y 的逐元素乘积
        def h(x, y):
            return y.sum(), x * y

        # 生成一个标量张量 x
        x = ctors.randn([])
        # 生成一个大小为 (1,) 的随机张量 y
        y = ctors.randn(1)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 h 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(h, (x, y))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_different_devices(self, ctors):
        # 定义函数 f，接受两个输入 x 和 y，返回两个输出：x 与 y 的逐元素乘积，以及 x 与 y 的逐元素乘积在 CUDA 上的张量
        def f(x, y):
            return x * y, (x * y).cuda()

        # 生成一个大小为 (3,) 的随机张量 x
        x = ctors.randn(3)
        # 生成一个大小为 (3,) 的随机张量 y
        y = ctors.randn(3)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性
        self._check_jacobian_vectorize_correctness(f, (x, y))

    @base_and_logging_tensor
    def test_jacobian_vectorize_correctness_different_dtype(self, ctors):
        # 定义函数 f，接受两个输入 x 和 y，返回两个输出：x 与 y 的逐元素乘积，分别转换为 float 和 double 类型
        def f(x, y):
            return (x * y).float(), (x * y).double()

        # 生成一个大小为 (3,) 的随机张量 x
        x = ctors.randn(3)
        # 生成一个大小为 (3,) 的随机张量 y
        y = ctors.randn(3)
        # 使用 _check_jacobian_vectorize_correctness 函数检验函数 f 的雅可比向量化正确性，并关闭测试前向自动求导
        self._check_jacobian_vectorize_correctness(f, (x, y), test_forward_ad=False)

    def _check_hessian_vectorize_correctness(self, f, inputs):
        # 计算函数 f 在输入 inputs 上的 Hessian 矩阵，向量化为 True
        expected = autogradF.hessian(f, inputs, vectorize=False)
        result = autogradF.hessian(f, inputs, vectorize=True)
        # 检查向量化计算得到的 Hessian 矩阵结果是否与非向量化计算结果一致
        self.assertEqual(result, expected)

        # 使用前向模式计算 Hessian 矩阵的结果，向量化为 True
        result_forward_mode = autogradF.hessian(
            f, inputs, outer_jacobian_strategy="forward-mode", vectorize=True
        )
        # 检查前向模式计算得到的 Hessian 矩阵结果是否与非向量化计算结果一致
        self.assertEqual(result_forward_mode, expected)
    @base_and_logging_tensor



# 将当前测试函数装饰为基础张量操作并记录日志
def test_hessian_vectorize_correctness_simple(self, ctors):
    # 定义一个简单的函数 f(x)，计算 3 * x^2 的和
    def f(x):
        return (3 * x**2).sum()

    # 生成一个指定形状的随机张量 x
    x = ctors.randn(2, 3, 5)
    # 调用私有方法 _check_hessian_vectorize_correctness，验证 Hessian 向量化的正确性
    self._check_hessian_vectorize_correctness(f, x)

@base_and_logging_tensor
def test_hessian_vectorize_correctness_multi_input(self, ctors):
    # 定义一个多输入函数 f(x, y, z)，计算 ((x.relu() * x) @ y.sin() @ z).sum()
    def f(x, y, z):
        return ((x.relu() * x) @ y.sin() @ z).sum()

    # 生成指定形状的随机张量 x, y, z
    x = ctors.randn(2, 3)
    y = ctors.randn(3, 5)
    z = ctors.randn(5, 5)
    # 调用私有方法 _check_hessian_vectorize_correctness，验证 Hessian 向量化的正确性
    self._check_hessian_vectorize_correctness(f, (x, y, z))

@base_and_logging_tensor
def test_hessian_vectorize_correctness_unrelated_outputs(self, ctors):
    # 定义一个函数 f(x, y)，计算 (x**2).sum()
    def f(x, y):
        return (x**2).sum()

    # 生成指定形状的随机张量 x, y
    x = ctors.randn(2)
    y = ctors.randn(3)
    # 调用私有方法 _check_hessian_vectorize_correctness，验证 Hessian 向量化的正确性
    self._check_hessian_vectorize_correctness(f, (x, y))

    # 定义一个函数 f(x, y)，返回一个标量张量
    def f(x, y):
        return ctors.ones([])

    # 生成指定形状的随机张量 x, y
    x = ctors.randn(2)
    y = ctors.randn(3)
    # 调用私有方法 _check_hessian_vectorize_correctness，验证 Hessian 向量化的正确性
    self._check_hessian_vectorize_correctness(f, (x, y))

@parametrize("vectorize", [True, False])
@base_and_logging_tensor
def test_hessian_err_check(self, vectorize, ctors):
    # 定义一个函数 foo(a)，计算 3 * a.narrow(0, 0, 3).exp().sum()
    def foo(a):
        return 3 * a.narrow(0, 0, 3).exp().sum()

    # 定义一个函数 bar(a)，计算 3 * a.narrow(0, 0, 3)，同时返回字符串 "bar"
    def bar(a):
        return 3 * a.narrow(0, 0, 3), "bar"

    # 定义一个函数 bar2(a)，计算 3 * a.narrow(0, 0, 3)，预期返回单个张量而非元组
    def bar2(a):
        return 3 * a.narrow(0, 0, 3)

    # 定义一个函数 bar3(a)，计算 3 * a.narrow(0, 0, 3)，预期返回单个张量而非元组
    def bar3(a):
        return 3 * a.narrow(0, 0, 3), 3 * a.narrow(0, 0, 3)

    # 生成一个随机张量 inp
    inp = ctors.rand(4)
    # 使用断言确保在调用 autogradF.hessian(foo, (inp, 2), vectorize=vectorize) 时出现 TypeError 异常
    with self.assertRaisesRegex(
        TypeError, "The inputs given to hessian must be either a Tensor"
    ):
        res = autogradF.hessian(foo, (inp, 2), vectorize=vectorize)

    # 使用断言确保在调用 autogradF.hessian(bar, inp, vectorize=vectorize) 时出现 TypeError 异常
    with self.assertRaisesRegex(
        TypeError, "The outputs of the user-provided function given to hessian must"
    ):
        res = autogradF.hessian(bar, inp, vectorize=vectorize)

    # 定义错误消息字符串，用于在断言时进行比较
    err_msg_out = "The Tensor returned by the function given to hessian should contain a single element"
    # 使用断言确保在调用 autogradF.hessian(bar2, inp, vectorize=vectorize) 时出现 RuntimeError 异常
    with self.assertRaisesRegex(RuntimeError, err_msg_out):
        res = autogradF.hessian(bar2, inp, vectorize=vectorize)

    # 使用断言确保在调用 autogradF.hessian(bar3, inp, vectorize=vectorize) 时出现 RuntimeError 异常
    with self.assertRaisesRegex(
        RuntimeError, "The function given to hessian should return a single Tensor"
    ):
        res = autogradF.hessian(bar3, inp, vectorize=vectorize)

    # 调用 autogradF.hessian(foo, inp, vectorize=vectorize)，并使用私有方法 _assert_interleaved_struct 进行验证
    res = autogradF.hessian(foo, inp, vectorize=vectorize)
    self._assert_interleaved_struct(res, inp, inp)

    # 重新定义函数 foo(a, b)，计算 (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()
    def foo(a, b):
        return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

    # 生成随机张量 inp
    inp = (ctors.rand(4), ctors.rand(5))
    # 调用 autogradF.hessian(foo, inp, vectorize=vectorize)，并使用私有方法 _assert_interleaved_struct 进行验证
    res = autogradF.hessian(foo, inp, vectorize=vectorize)
    self._assert_interleaved_struct(res, inp, inp)
    # 定义一个测试函数，用于检查 Hessian 矩阵严格模式下的异常处理
    def test_hessian_err_check_strict(self, ctors):
        # 内部函数 foo，计算输入张量的所有元素之和
        def foo(a):
            return a.detach().sum()

        # 内部函数 bar，创建一个不与输入连接的、需要梯度的非叶子张量，并对其进行操作
        def bar(a):
            return a.long().float().requires_grad_().clone().sum()

        # 内部函数 bar2，对输入进行线性变换，此变换的雅可比矩阵与输入无关
        def bar2(a):
            return (3 * a).sum()

        # 生成一个随机输入张量
        inp = ctors.rand(4)

        # 断言在严格模式下调用 foo 函数的 Hessian 方法会抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.hessian(foo, inp, strict=True)
        
        # 在非严格模式下调用 foo 函数的 Hessian 方法，返回结果 res
        res = autogradF.hessian(foo, inp, strict=False)
        # 断言 res 的绝对值之和为 0.0
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

        # 断言在严格模式下调用 bar 函数的 Hessian 方法会抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0",
        ):
            res = autogradF.hessian(bar, inp, strict=True)
        
        # 在非严格模式下调用 bar 函数的 Hessian 方法，返回结果 res
        res = autogradF.hessian(bar, inp, strict=False)
        # 断言 res 的绝对值之和为 0.0
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

        # 断言在严格模式下调用 bar2 函数的 Hessian 方法会抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.hessian(bar2, inp, strict=True)
        
        # 在非严格模式下调用 bar2 函数的 Hessian 方法，返回结果 res
        res = autogradF.hessian(bar2, inp, strict=False)
        # 断言 res 的绝对值之和为 0.0
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.0)

    # 用装饰器 base_and_logging_tensor 修饰的向量化测试函数，检查 Hessian 矩阵严格模式下的异常处理
    @base_and_logging_tensor
    def test_hessian_err_check_strict_vectorize(self, ctors):
        # 内部函数 foo，计算输入张量元素的立方和
        def foo(x):
            return (x**3).sum()

        # 生成一个随机输入张量
        inp = ctors.rand(4)

        # 断言调用 foo 函数的 Hessian 方法在严格模式和向量化的情况下会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.hessian(foo, inp, strict=True, vectorize=True)

    # 用装饰器 base_and_logging_tensor 修饰的测试函数，检查 Hessian 矩阵在没有梯度下降时的情况
    def test_hessian_no_grad(self, ctors):
        # 内部函数 pow_reducer，计算输入张量元素的三次方和
        def pow_reducer(x):
            return x.pow(3).sum()

        # 生成一个 2x2 的随机输入张量
        inputs = ctors.rand(2, 2)

        # 使用 torch.no_grad 上下文，调用 pow_reducer 函数的 Hessian 方法
        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs)
        # 断言 res 中的每个元素的 grad_fn 属性为 None
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)
        # 断言 res 不等于一个全零的 2x2x2 张量
        self.assertNotEqual(res, ctors.zeros(2, 2, 2))

        # 使用 torch.no_grad 上下文，调用 pow_reducer 函数的 Hessian 方法，同时创建计算图
        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs, create_graph=True)
        # 断言 res 中的每个元素的 grad_fn 属性不为 None
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)
        # 断言 res 不等于一个全零的 2x2x2 张量
        self.assertNotEqual(res, ctors.zeros(2, 2, 2))
    # 定义一个测试方法，用于测试 autogradF.hessian 函数的输出
    def test_hessian_output(self, vectorize, ctors):
        # 定义一个函数 pow_reducer，对输入的张量进行立方求和操作
        def pow_reducer(x):
            return x.pow(3).sum()

        # 生成一个随机的 2x2 张量作为输入
        inputs = ctors.rand(2, 2)
        # 调用 autogradF.hessian 函数计算 pow_reducer 的 Hessian 矩阵
        res = autogradF.hessian(pow_reducer, inputs, vectorize=vectorize)
        # 断言计算得到的结果符合特定的结构要求
        self._assert_interleaved_struct(res, inputs, inputs)
        # 断言结果的梯度函数为 None
        self.assertIsNone(res.grad_fn)

        # 定义一个函数 add_pow_reducer，对输入的两个张量进行加法后立方求和操作
        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        # 生成两个随机的 2x2 张量作为输入
        inputs = (ctors.rand(2, 2), ctors.rand(2, 2))
        # 调用 autogradF.hessian 函数计算 add_pow_reducer 的 Hessian 矩阵
        res = autogradF.hessian(add_pow_reducer, inputs, vectorize=vectorize)
        # 断言计算得到的结果符合特定的结构要求
        self._assert_interleaved_struct(res, inputs, inputs)
        # 断言结果的梯度函数为 None
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)

    # 使用 parametrize 装饰器，对 test_hessian_scalar 方法进行多次参数化测试
    @parametrize("vectorize", [True, False])
    @base_and_logging_tensor
    # 定义一个测试方法，用于测试 autogradF.hessian 函数在标量输入下的行为
    def test_hessian_scalar(self, vectorize, ctors):
        # 定义一个简单的求和函数 reducer
        def reducer(x):
            return x.sum()

        # 生成一个随机的 4x4 张量作为输入
        inputs = ctors.rand(4, 4)
        # 调用 autogradF.hessian 函数计算 reducer 的 Hessian 矩阵
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        # 断言计算得到的结果符合特定的结构要求
        self._assert_interleaved_struct(res, inputs, inputs)

        # 生成一个随机的标量张量作为输入
        inputs = ctors.rand([])
        # 调用 autogradF.hessian 函数计算 reducer 的 Hessian 矩阵
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        # 断言计算得到的结果符合特定的结构要求
        self._assert_same_struct(res, inputs)

        # 定义一个不符合要求的函数 bad_reducer，对输入的张量进行求和后尝试调整形状
        def bad_reducer(x):
            return x.sum().view(1, 1, 1)

        # 生成一个随机的 4x4 张量作为输入
        inputs = ctors.rand(4, 4)
        # 调用 autogradF.hessian 函数计算 bad_reducer 的 Hessian 矩阵
        res = autogradF.hessian(bad_reducer, inputs, vectorize=vectorize)
        # 断言计算得到的结果符合特定的结构要求
        self._assert_interleaved_struct(res, inputs, inputs)
    # 定义一个测试方法，用于测试创建带有图形的 Hessian 矩阵
    def test_hessian_create_graph(self, vectorize, ctors):
        # 定义一个函数 pow_reducer，对输入的张量进行三次幂求和
        def pow_reducer(x):
            return x.pow(3).sum()

        # 生成一个随机张量输入，数据类型为双精度浮点型，要求梯度计算
        inputs = ctors.rand(2, 2, dtype=torch.double, requires_grad=True)
        # 计算 pow_reducer 函数的 Hessian 矩阵，同时创建计算图
        res = autogradF.hessian(
            pow_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        # 断言 Hessian 矩阵的结构正确性，输入与输出张量应该一致
        self._assert_interleaved_struct(res, inputs, inputs)
        # 确保计算结果具有梯度函数
        self.assertIsNotNone(res.grad_fn)

        # 对 pow_reducer 函数进行梯度检查
        gradcheck(
            lambda inp: autogradF.hessian(
                pow_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )
        # 对 pow_reducer 函数的二阶梯度进行检查
        gradgradcheck(
            lambda inp: autogradF.hessian(
                pow_reducer, inp, create_graph=True, vectorize=vectorize
            ),
            inputs,
        )

        # 定义一个新的函数 add_pow_reducer，对两个输入的和进行三次幂求和
        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        # 生成两个随机张量输入，数据类型为双精度浮点型，要求梯度计算
        inputs = (
            ctors.rand(2, 2, dtype=torch.double, requires_grad=True),
            ctors.rand(2, 2, dtype=torch.double, requires_grad=True),
        )
        # 计算 add_pow_reducer 函数的 Hessian 矩阵，同时创建计算图
        res = autogradF.hessian(
            add_pow_reducer, inputs, create_graph=True, vectorize=vectorize
        )
        # 断言 Hessian 矩阵的结构正确性，输入与输出张量应该一致
        self._assert_interleaved_struct(res, inputs, inputs)
        # 确保计算结果的各个部分都具有梯度函数
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)

        # 定义一个函数 flatten，用于将多层嵌套的元组扁平化
        def flatten(inp):
            return tuple(el_lvl2 for el_lvl1 in inp for el_lvl2 in el_lvl1)

        # 对 add_pow_reducer 函数进行梯度检查，同时扁平化输出结果
        gradcheck(
            lambda *inp: flatten(
                autogradF.hessian(
                    add_pow_reducer, inp, create_graph=True, vectorize=vectorize
                )
            ),
            inputs,
        )
        # 对 add_pow_reducer 函数的二阶梯度进行检查，同时扁平化输出结果
        gradgradcheck(
            lambda *inp: flatten(
                autogradF.hessian(
                    add_pow_reducer, inp, create_graph=True, vectorize=vectorize
                )
            ),
            inputs,
        )

        # 定义一个函数 foo，对输入张量进行余弦函数运算，并计算其 Hessian 矩阵
        def foo(x, y):
            x = x.cos()
            # 计算 add_pow_reducer 函数在余弦变换后的值以及其 Hessian 矩阵
            val, hess = autogradF.hessian(
                add_pow_reducer, (x, y), create_graph=True, vectorize=vectorize
            )

            # 计算结果包括对值的余弦求和以及 Hessian 矩阵的余弦求和
            res = val[0].cos().sum() + val[1].cos().sum() + hess[0].cos().sum()
            res = res + hess[1].cos().sum() + x.cos().sum() + y.cos().sum()
            return res

        # 对 foo 函数进行梯度检查
        gradcheck(foo, inputs)
        # 对 foo 函数的二阶梯度进行检查
        gradgradcheck(foo, inputs)
    # 定义一个测试方法，用于检查在给定构造器（ctors）下的错误情况
    def test_vhp_err_check(self, ctors):
        # 定义一个内部函数 foo，对输入张量进行操作并返回结果
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 定义一个内部函数 bar，对输入张量进行操作并返回结果和字符串 "bar"
        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        # 定义一个内部函数 bar2，对输入张量进行操作并返回结果
        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        # 生成一个随机张量作为输入 inp 和 v
        inp = ctors.rand(4)
        v = ctors.rand(4)

        # 使用断言检查 autogradF.vhp 函数对 foo 的调用是否会引发 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "The inputs given to vhp must be either a Tensor"
        ):
            res = autogradF.vhp(foo, (inp, 2), v)

        # 使用断言检查 autogradF.vhp 函数对 bar 的调用是否会引发 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to vhp must"
        ):
            res = autogradF.vhp(bar, inp, v)

        # 检查 autogradF.vhp 函数对 bar2 的调用是否会引发 RuntimeError 异常，错误信息包含特定内容
        err_msg_out = "The Tensor returned by the function given to vhp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.vhp(bar2, inp, v)

        # 检查 autogradF.vhp 函数对 foo 的调用是否会引发 RuntimeError 异常，错误信息包含特定内容
        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.vhp(foo, inp, ctors.rand(5))

        # 使用断言检查 autogradF.vhp 函数对 foo 的调用是否会引发 TypeError 异常
        with self.assertRaisesRegex(
            TypeError,
            "The v given to vhp must be either a Tensor or a tuple of Tensors",
        ):
            res = autogradF.vhp(foo, inp, (v, 2))

        # 正常调用 autogradF.vhp 函数，获取结果并使用内部方法 _assert_same_struct 检查结构是否一致
        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        # 定义一个新的 foo 函数，接受两个参数并对它们进行操作后返回结果
        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        # 生成两个随机张量作为输入 inp 和 v
        inp = (ctors.rand(4), ctors.rand(5))
        v = (ctors.rand(4), ctors.rand(5))

        # 调用 autogradF.vhp 函数，获取结果并使用内部方法 _assert_same_struct 检查结构是否一致
        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    # 标记为 base_and_logging_tensor 的装饰器
    @base_and_logging_tensor
    # 定义一个测试方法，用于检查严格模式下 autogradF.vhp 方法的错误处理
    def test_vhp_err_check_strict(self, ctors):
        # 定义一个简单的函数 foo，对输入进行操作并返回结果的总和
        def foo(a):
            return a.detach().sum()

        # 定义一个复杂的函数 bar，创建一个不是叶子节点但需要梯度的 Tensor，并对其进行操作后返回总和
        def bar(a):
            # 创建一个不是叶子节点的 Tensor，需要梯度，但与输入没有关联
            return a.long().float().requires_grad_().clone().sum()

        # 定义另一个函数 bar2，这个函数是一个线性函数，其雅可比矩阵与输入无关
        def bar2(a):
            return (3 * a).sum()

        # 生成一个随机输入向量 inp 和一个随机向量 v
        inp = ctors.rand(4)
        v = ctors.rand(4)

        # 测试 foo 函数，期望在严格模式下出现 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.vhp(foo, inp, v, strict=True)

        # 测试 foo 函数，在非严格模式下计算 vhp 结果，并进行结果的结构和值的断言
        res = autogradF.vhp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 测试 bar 函数，期望在严格模式下出现 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.vhp(bar, inp, v, strict=True)

        # 测试 bar 函数，在非严格模式下计算 vhp 结果，并进行结果的结构和值的断言
        res = autogradF.vhp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 测试 bar2 函数，期望在严格模式下出现 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.vhp(bar2, inp, v, strict=True)

        # 测试 bar2 函数，在非严格模式下计算 vhp 结果，并进行结果的结构和值的断言
        res = autogradF.vhp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.0)
    # 定义一个测试方法，用于测试自动求导库中的 vhp 函数对不同函数的输出
    def test_vhp_output(self, ctors):
        # 定义一个简单的函数 foo，对输入的张量进行操作并返回结果
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 生成输入张量
        inputs = ctors.rand(4, 4)
        # 创建一个全为 1 的张量
        v = ctors.ones(4, 4)
        # 使用 autogradF.vhp 函数计算函数 foo 对 inputs 和 v 的导数信息
        res = autogradF.vhp(foo, inputs, v)
        # 断言 res[1] 的结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)
        # 断言返回的导数信息中没有梯度函数（grad_fn）
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        # 定义另一个函数 bar，接受两个输入张量，并对其进行操作
        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        # 生成不同形状的输入张量和对应的全为 1 的张量
        inputs = (ctors.rand(3), ctors.rand(4))
        v = (ctors.ones(3), ctors.ones(4))
        # 使用 autogradF.vhp 函数计算函数 bar 对 inputs 和 v 的导数信息
        out, vhp_val = autogradF.vhp(bar, inputs, v)
        # 断言 vhp_val 的结构与 inputs 相同
        self._assert_same_struct(vhp_val, inputs)
        # 断言返回的导数信息中没有梯度函数（grad_fn）
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vhp_val[0].grad_fn)
        self.assertIsNone(vhp_val[1].grad_fn)

    @base_and_logging_tensor
    # 标记为 base_and_logging_tensor 的测试方法，用于测试特定张量操作的自动求导功能
    def test_vhp_scalar(self, ctors):
        # 定义一个简单的求和函数 reducer，用于测试标量情况下的自动求导
        def reducer(x):
            return x.sum()

        # 生成不同形状的输入张量和对应的全为 1 的张量
        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        # 使用 autogradF.vhp 函数计算函数 reducer 对 inputs 和 v 的导数信息
        res = autogradF.vhp(reducer, inputs, v)
        # 断言 res[1] 的结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 生成标量输入张量和对应的随机张量
        inputs = ctors.rand([])
        v = ctors.rand([])
        # 使用 autogradF.vhp 函数计算函数 reducer 对 inputs 和 v 的导数信息
        res = autogradF.vhp(reducer, inputs, v)
        # 断言 res[1] 的结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 生成仅有输入张量的标量输入情况
        res = autogradF.vhp(reducer, inputs)
        # 断言 res[1] 的结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 定义一个返回非标量张量的函数 bad_reducer，用于测试其对自动求导的影响
        def bad_reducer(x):
            return x.sum().view(1, 1, 1)

        # 生成不同形状的输入张量和对应的随机张量
        inputs = ctors.rand(4, 4)
        v = ctors.rand(4, 4)
        # 使用 autogradF.vhp 函数计算函数 bad_reducer 对 inputs 和 v 的导数信息
        res = autogradF.vhp(bad_reducer, inputs, v)
        # 断言 res[1] 的结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)
    # 定义一个测试函数，用于测试 autogradF.vhp 函数的创建图功能
    def test_vhp_create_graph(self, ctors):
        # 定义一个简单的函数 foo(a)，计算输入张量 a 的前三个元素指数函数的和乘以3
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 创建一个随机张量作为输入，数据类型为双精度浮点数，同时需要计算梯度
        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        # 创建一个全为1的张量作为 v，数据类型为双精度浮点数，同时需要计算梯度
        v = ctors.ones(4, 4, dtype=torch.double, requires_grad=True)
        # 调用 autogradF.vhp 函数计算 foo 函数对 inputs 和 v 的偏导数，同时创建计算图
        res = autogradF.vhp(foo, inputs, v, create_graph=True)
        # 断言 res 的第一个返回值与 inputs 结构相同
        self._assert_same_struct(res[1], inputs)
        # 断言 res 的第一个返回值存在梯度函数
        self.assertIsNotNone(res[0].grad_fn)
        # 断言 res 的第二个返回值存在梯度函数
        self.assertIsNotNone(res[1].grad_fn)

        # 使用 gradcheck 函数验证 autogradF.vhp 函数的梯度
        gradcheck(
            lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v)
        )
        # 使用 gradgradcheck 函数验证 autogradF.vhp 函数的二阶梯度
        gradgradcheck(
            lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v)
        )

        # 定义另一个函数 bar(a, b)，计算 a 加上 b 的前三个元素乘以3的指数函数的和
        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        # 创建两个输入张量，分别为长度为3和4的随机张量，数据类型为双精度浮点数，同时需要计算梯度
        inputs = (
            ctors.rand(3, dtype=torch.double, requires_grad=True),
            ctors.rand(4, dtype=torch.double, requires_grad=True),
        )
        # 创建两个全为1的张量作为 v，分别对应上述的两个输入张量，数据类型为双精度浮点数，同时需要计算梯度
        v = (
            ctors.ones(3, dtype=torch.double, requires_grad=True),
            ctors.ones(4, dtype=torch.double, requires_grad=True),
        )
        # 调用 autogradF.vhp 函数计算 bar 函数对 inputs 和 v 的偏导数，同时创建计算图
        out, vhp_val = autogradF.vhp(bar, inputs, v, create_graph=True)
        # 断言 vhp_val 的结构与 inputs 相同
        self._assert_same_struct(vhp_val, inputs)
        # 断言 out 的梯度函数存在
        self.assertIsNotNone(out.grad_fn)
        # 断言 vhp_val 的第一个返回值的梯度函数存在
        self.assertIsNotNone(vhp_val[0].grad_fn)
        # 断言 vhp_val 的第二个返回值的梯度函数存在
        self.assertIsNotNone(vhp_val[1].grad_fn)

        # 使用 gradcheck 函数验证 autogradF.vhp 函数对 bar 函数的偏导数
        gradcheck(
            lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )
        # 使用 gradgradcheck 函数验证 autogradF.vhp 函数对 bar 函数的二阶偏导数
        gradgradcheck(
            lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        # 定义一个函数 foo(*args)，其中 args 包含两个输入 x 和 y，以及一个 v 的列表
        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            # 对 x 应用余弦函数
            x = x.cos()
            # 调用 autogradF.vhp 函数计算 bar 函数对 (x, y) 和 v 的偏导数，同时创建计算图
            val, grad = autogradF.vhp(bar, (x, y), v, create_graph=True)

            # 返回值为 val 的余弦函数加上各部分的余弦函数和
            return (
                val.cos()
                + grad[0].cos().sum()
                + grad[1].cos()
                + x.cos().sum()
                + y.cos()
            )

        # 使用 gradcheck 函数验证 foo 函数的梯度
        gradcheck(foo, inputs + v)
        # 使用 gradgradcheck 函数验证 foo 函数的二阶梯度
        gradgradcheck(foo, inputs + v)

    @base_and_logging_tensor
    # 定义一个测试函数，用于检查 autogradF.hvp 函数的错误处理功能
    def test_hvp_err_check(self, ctors):
        
        # 定义一个简单的函数 foo，用于测试 autogradF.hvp 的正确调用
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()
        
        # 定义函数 bar，返回一个元组，用于测试 autogradF.hvp 对输出类型的要求
        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"
        
        # 定义函数 bar2，返回一个张量，但不符合 autogradF.hvp 的输出要求
        def bar2(a):
            return 3 * a.narrow(0, 0, 3)
        
        # 生成随机输入张量 inp 和 v
        inp = ctors.rand(4)
        v = ctors.rand(4)
        
        # 测试 autogradF.hvp 对输入参数的类型错误处理，期望引发 TypeError 异常
        res = autogradF.hvp(foo, inp, v)
        with self.assertRaisesRegex(
            TypeError, "The inputs given to hvp must be either a Tensor"
        ):
            res = autogradF.hvp(foo, (inp, 2), v)
        
        # 测试 autogradF.hvp 对输出类型错误处理，期望引发 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "The outputs of the user-provided function given to hvp must"
        ):
            res = autogradF.hvp(bar, inp, v)
        
        # 测试 autogradF.hvp 对输出张量尺寸的错误处理，期望引发 RuntimeError 异常
        err_msg_out = "The Tensor returned by the function given to hvp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hvp(bar2, inp, v)
        
        # 测试 autogradF.hvp 对 v 张量尺寸错误处理，期望引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.hvp(foo, inp, ctors.rand(5))
        
        # 测试 autogradF.hvp 对 v 参数类型错误处理，期望引发 TypeError 异常
        with self.assertRaisesRegex(
            TypeError,
            "The v given to hvp must be either a Tensor or a tuple of Tensors",
        ):
            res = autogradF.hvp(foo, inp, (v, 2))
        
        # 再次使用正确的参数调用 autogradF.hvp，检查返回结果结构是否符合预期
        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)
        
        # 重新定义函数 foo，接受两个参数，用于测试 autogradF.hvp 的多参数支持
        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()
        
        # 生成两个随机输入元组 inp 和 v，分别包含不同长度的张量
        inp = (ctors.rand(4), ctors.rand(5))
        v = (ctors.rand(4), ctors.rand(5))
        
        # 调用 autogradF.hvp，检查返回结果结构是否符合预期
        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    @base_and_logging_tensor
    # 定义一个测试方法，用于检查 Hessian-Vector Product 的严格模式下的错误处理
    def test_hvp_err_check_strict(self, ctors):
        # 定义一个简单的函数 foo，对输入进行操作并返回其元素之和
        def foo(a):
            return a.detach().sum()

        # 定义一个复杂一点的函数 bar，创建一个非叶节点张量，要求梯度但与输入无关
        def bar(a):
            return a.long().float().requires_grad_().clone().sum()

        # 定义函数 bar2，一个线性函数，其雅可比矩阵与输入无关
        def bar2(a):
            return (3 * a).sum()

        # 生成随机输入张量 inp 和 v
        inp = ctors.rand(4)
        v = ctors.rand(4)

        # 测试 foo 函数在严格模式下调用 hvp 函数，预期抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of the user-provided function does not require gradients.",
        ):
            res = autogradF.hvp(foo, inp, v, strict=True)

        # 调用 hvp 函数计算 foo 函数的 Hessian-Vector Product，非严格模式
        res = autogradF.hvp(foo, inp, v, strict=False)
        # 断言返回结果的结构与输入 inp 相同
        self._assert_same_struct(res[1], inp)
        # 断言返回结果的绝对值之和为 0
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 测试 bar 函数在严格模式下调用 hvp 函数，预期抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "The output of the user-provided function is independent of input 0",
        ):
            res = autogradF.hvp(bar, inp, v, strict=True)

        # 调用 hvp 函数计算 bar 函数的 Hessian-Vector Product，非严格模式
        res = autogradF.hvp(bar, inp, v, strict=False)
        # 断言返回结果的结构与输入 inp 相同
        self._assert_same_struct(res[1], inp)
        # 断言返回结果的绝对值之和为 0
        self.assertEqual(res[1].abs().sum(), 0.0)

        # 测试 bar2 函数在严格模式下调用 hvp 函数，预期抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "jacobian of the user-provided function with respect to input 0 is",
        ):
            res = autogradF.hvp(bar2, inp, v, strict=True)

        # 调用 hvp 函数计算 bar2 函数的 Hessian-Vector Product，非严格模式
        res = autogradF.hvp(bar2, inp, v, strict=False)
        # 断言返回结果的结构与输入 inp 相同
        self._assert_same_struct(res[1], inp)
        # 断言返回结果的绝对值之和为 0
        self.assertEqual(res[1].abs().sum(), 0.0)
    # 定义测试方法，用于验证 autogradF.hvp 函数的输出结果
    def test_hvp_output(self, ctors):
        # 内部定义函数 foo，接受参数 a，返回对应操作的结果
        def foo(a):
            # 对输入张量 a 进行切片操作，计算指定范围内的指数和
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 使用 ctors 生成随机输入张量 inputs
        inputs = ctors.rand(4, 4)
        # 创建全为1的张量 v
        v = ctors.ones(4, 4)
        # 计算函数 foo 对于 inputs 和 v 的 Hessian-Vector Product (HVP)
        res = autogradF.hvp(foo, inputs, v)
        # 断言 res 的第一个元素结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)
        # 断言 res 的第一个元素的梯度函数为空
        self.assertIsNone(res[0].grad_fn)
        # 断言 res 的第二个元素的梯度函数为空
        self.assertIsNone(res[1].grad_fn)

        # 内部定义函数 bar，接受参数 a 和 b，返回对应操作的结果
        def bar(a, b):
            # 对输入张量 a 和 b 进行指定范围内的加权和、指数和操作
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        # 使用 ctors 生成随机输入元组 inputs
        inputs = (ctors.rand(3), ctors.rand(4))
        # 创建全为1的元组 v
        v = (ctors.ones(3), ctors.ones(4))
        # 计算函数 bar 对于 inputs 和 v 的 HVP，并返回结果和 hvp_val
        out, hvp_val = autogradF.hvp(bar, inputs, v)
        # 断言 hvp_val 的结构与 inputs 相同
        self._assert_same_struct(hvp_val, inputs)
        # 断言 out 的梯度函数为空
        self.assertIsNone(out.grad_fn)
        # 断言 hvp_val 中的第一个元素的梯度函数为空
        self.assertIsNone(hvp_val[0].grad_fn)
        # 断言 hvp_val 中的第二个元素的梯度函数为空
        self.assertIsNone(hvp_val[1].grad_fn)

    # 使用 base_and_logging_tensor 装饰器定义测试方法，验证 autogradF.hvp 函数对标量的处理
    @base_and_logging_tensor
    def test_hvp_scalar(self, ctors):
        # 内部定义函数 reducer，接受参数 x，返回对应操作的结果
        def reducer(x):
            # 计算输入张量 x 的指数和
            return x.exp().sum()

        # 使用 ctors 生成随机输入张量 inputs
        inputs = ctors.rand(4, 4)
        # 创建全为1的张量 v
        v = ctors.ones(4, 4)
        # 计算函数 reducer 对于 inputs 和 v 的 HVP
        res = autogradF.hvp(reducer, inputs, v)
        # 断言 res 的第一个元素结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 使用 ctors 生成随机标量输入 inputs
        inputs = ctors.rand([])
        # 使用 ctors 生成随机标量输入 v
        v = ctors.rand([])
        # 计算函数 reducer 对于 inputs 和 v 的 HVP
        res = autogradF.hvp(reducer, inputs, v)
        # 断言 res 的第一个元素结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 计算函数 reducer 对于 inputs 的 HVP
        res = autogradF.hvp(reducer, inputs)
        # 断言 res 的第一个元素结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)

        # 内部定义函数 bad_reducer，接受参数 x，返回对应操作的结果
        def bad_reducer(x):
            # 计算输入张量 x 的指数和，并将结果视图转换为 (1, 1, 1) 形状
            return x.exp().sum().view(1, 1, 1)

        # 使用 ctors 生成随机输入张量 inputs
        inputs = ctors.rand(4, 4)
        # 使用 ctors 生成随机输入张量 v
        v = ctors.rand(4, 4)
        # 计算函数 bad_reducer 对于 inputs 和 v 的 HVP
        res = autogradF.hvp(bad_reducer, inputs, v)
        # 断言 res 的第一个元素结构与 inputs 相同
        self._assert_same_struct(res[1], inputs)
    # 定义一个测试函数，用于测试 autogradF.hvp 的功能
    def test_hvp_create_graph(self, ctors):
        # 定义一个简单的函数 foo，对输入进行操作并返回结果
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 创建输入张量 inputs，随机初始化，数据类型为双精度浮点型，需要梯度信息
        inputs = ctors.rand(4, 4, dtype=torch.double, requires_grad=True)
        # 创建张量 v，所有元素为 1，数据类型为双精度浮点型，需要梯度信息
        v = ctors.ones(4, 4, dtype=torch.double, requires_grad=True)
        # 计算函数 foo 在 inputs 和 v 上的 Hessian 向量积，返回值需要创建计算图
        res = autogradF.hvp(foo, inputs, v, create_graph=True)
        # 断言返回值的结构与 inputs 张量相同
        self._assert_same_struct(res[1], inputs)
        # 断言返回值的第一个元素具有梯度函数
        self.assertIsNotNone(res[0].grad_fn)
        # 断言返回值的第二个元素具有梯度函数
        self.assertIsNotNone(res[1].grad_fn)

        # 使用 gradcheck 函数验证函数 foo 在 inputs 和 v 上的 Hessian 向量积
        gradcheck(
            lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v)
        )
        # 使用 gradgradcheck 函数验证函数 foo 在 inputs 和 v 上的 Hessian 向量积
        gradgradcheck(
            lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v)
        )

        # 定义一个带有两个参数的函数 bar
        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        # 创建输入元组 inputs，其中包含两个张量，随机初始化，数据类型为双精度浮点型，需要梯度信息
        inputs = (
            ctors.rand(3, dtype=torch.double, requires_grad=True),
            ctors.rand(4, dtype=torch.double, requires_grad=True),
        )
        # 创建张量 v，其中包含两个元素，所有元素为 1，数据类型为双精度浮点型，需要梯度信息
        v = (
            ctors.ones(3, dtype=torch.double, requires_grad=True),
            ctors.ones(4, dtype=torch.double, requires_grad=True),
        )
        # 计算函数 bar 在 inputs 和 v 上的 Hessian 向量积，并返回输出和 Hessian 向量积的值，需要创建计算图
        out, hvp_val = autogradF.hvp(bar, inputs, v, create_graph=True)
        # 断言返回值 hvp_val 的结构与 inputs 元组相同
        self._assert_same_struct(hvp_val, inputs)
        # 断言输出 out 具有梯度函数
        self.assertIsNotNone(out.grad_fn)
        # 断言 hvp_val 的第一个元素具有梯度函数
        self.assertIsNotNone(hvp_val[0].grad_fn)
        # 断言 hvp_val 的第二个元素具有梯度函数
        self.assertIsNotNone(hvp_val[1].grad_fn)

        # 使用 gradcheck 函数验证函数 bar 在 inputs 和 v 上的 Hessian 向量积，并返回其第二个元素
        gradcheck(
            lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )
        # 使用 gradgradcheck 函数验证函数 bar 在 inputs 和 v 上的 Hessian 向量积，并返回其第二个元素
        gradgradcheck(
            lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1],
            inputs + v,
        )

        # 定义一个可变参数函数 foo，接受任意数量的参数
        def foo(*args):
            # 将参数 args 解包为 x, y 和 v
            x, y = args[:2]
            v = args[2:]

            # 对 x 应用余弦函数
            x = x.cos()
            # 计算函数 bar 在 (x, y) 和 v 上的 Hessian 向量积，并返回其值和梯度
            val, grad = autogradF.hvp(bar, (x, y), v, create_graph=True)

            # 返回值为 x 的余弦值加上各部分的余弦值和
            return (
                val.cos()
                + grad[0].cos().sum()
                + grad[1].cos()
                + x.cos().sum()
                + y.cos()
            )

        # 使用 gradcheck 函数验证函数 foo 在 inputs 和 v 上的梯度
        gradcheck(foo, inputs + v)
        # 使用 gradgradcheck 函数验证函数 foo 在 inputs 和 v 上的梯度
        gradgradcheck(foo, inputs + v)
    # 定义一个测试方法，用于测试 Hessian 矩阵的匹配性
    def test_hessian_match_vhp_hvp(self, ctors):
        # 定义一个简单的函数 foo，对输入进行处理并返回一个标量值
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        # 使用给定的构造器生成一个随机输入向量
        inputs = ctors.rand(4)
        # 生成另一个随机向量作为 v
        v = ctors.rand(4)

        # 计算函数 foo 在 inputs 处的 Hessian 矩阵
        hes = autogradF.hessian(foo, inputs)
        # 计算 Hessian 矩阵 hes 与 v 的乘积的第二个部分，即 Hessian 矩阵和 v 的 Hessian 向量积
        hvp = autogradF.hvp(foo, inputs, v)[1]
        # 计算 v 和 Hessian 矩阵 hes 的乘积的第二部分，即 v 的 Hessian 向量积和 Hessian 矩阵
        vhp = autogradF.vhp(foo, inputs, v)[1]

        # 断言 hvp 应与 hes 与 v 的乘积相等
        self.assertEqual(hvp, torch.mm(hes, v.unsqueeze(1)).squeeze(1))
        # 断言 vhp 应与 v 与 hes 的乘积相等
        self.assertEqual(vhp, torch.mm(v.unsqueeze(0), hes).squeeze(0))
# 实例化带有参数化测试的 TestAutogradFunctional 类
instantiate_parametrized_tests(TestAutogradFunctional)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试
    run_tests()
```