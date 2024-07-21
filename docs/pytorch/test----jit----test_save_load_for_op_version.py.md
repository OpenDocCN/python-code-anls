# `.\pytorch\test\jit\test_save_load_for_op_version.py`

```
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import io
import os
import sys
from itertools import product as product  # 导入 product 函数并重命名为 product
from typing import Union  # 导入 Union 类型用于类型提示

import hypothesis.strategies as st  # 导入 hypothesis 库中的 strategies 模块，并重命名为 st
from hypothesis import example, given, settings  # 导入 hypothesis 库中的 example, given, settings 模块

import torch  # 导入 PyTorch 库

# 让 test/ 中的 helper 文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.jit.mobile import _load_for_lite_interpreter  # 从 torch.jit.mobile 中导入 _load_for_lite_interpreter 函数
from torch.testing._internal.jit_utils import JitTestCase  # 从 torch.testing._internal.jit_utils 中导入 JitTestCase 类

if __name__ == "__main__":
    # 如果当前脚本被直接运行，则抛出运行时错误，提醒使用正确的方式运行测试
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestSaveLoadForOpVersion(JitTestCase):
    # 辅助函数：保存和加载模块后返回模块对象
    def _save_load_module(self, m):
        scripted_module = torch.jit.script(m())  # 对输入的模块 m 进行脚本化
        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(scripted_module, buffer)  # 将脚本化后的模块保存到字节流缓冲区中
        buffer.seek(0)  # 将字节流缓冲区的指针位置移动到起始位置
        return torch.jit.load(buffer)  # 从字节流缓冲区中加载模块并返回

    # 辅助函数：保存和加载移动端模块后返回模块对象
    def _save_load_mobile_module(self, m):
        scripted_module = torch.jit.script(m())  # 对输入的模块 m 进行脚本化
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())  # 将脚本化后的模块保存为用于 Lite 解释器的字节流
        buffer.seek(0)  # 将字节流缓冲区的指针位置移动到起始位置
        return _load_for_lite_interpreter(buffer)  # 使用 Lite 解释器加载模块并返回

    # 辅助函数：执行一个函数并返回其结果或捕获的异常
    def _try_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)  # 执行函数 fn，并返回其结果
        except Exception as e:
            return e  # 捕获异常并返回异常对象

    # 辅助函数：验证图中特定类型的节点不存在
    def _verify_no(self, kind, m):
        self._verify_count(kind, m, 0)  # 调用 _verify_count 方法验证特定类型节点的数量为 0

    # 辅助函数：验证图中特定类型的节点数量符合预期
    def _verify_count(self, kind, m, count):
        # 统计图中特定类型节点的数量，并断言其与预期的数量 count 相等
        node_count = sum(str(n).count(kind) for n in m.graph.nodes())
        self.assertEqual(node_count, count)

    """
    Tests that verify Torchscript remaps aten::div(_) from versions 0-3
    to call either aten::true_divide(_), if an input is a float type,
    or truncated aten::divide(_) otherwise.
    NOTE: currently compares against current div behavior, too, since
      div behavior has not yet been updated.
    """

    @settings(
        max_examples=10, deadline=200000
    )  # 设置 hypothesis 测试的参数：最大生成例子数为 10，超时时间为 200000 微秒
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),  # 生成一个整数，范围在 [5, 199]
            st.floats(min_value=5.0, max_value=199.0),  # 生成一个浮点数，范围在 [5.0, 199.0]
        )
    )  # 使用 hypothesis 生成一个元组 (整数, 浮点数) 作为输入样本
    @example((2, 3, 2.0, 3.0))  # 添加一个例子：(2, 3, 2.0, 3.0)，确保这个例子被覆盖到
    # 测试版本化的张量除法函数
    def test_versioned_div_tensor(self, sample_input):
        # 定义一个用于历史版本除法的内部函数
        def historic_div(self, other):
            # 如果其中一个操作数是浮点数，则使用真实除法
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            # 否则使用截断模式的整数除法
            return self.divide(other, rounding_mode="trunc")

        # 定义一个继承自torch.nn.Module的自定义模块
        class MyModule(torch.nn.Module):
            # 模块的前向传播方法
            def forward(self, a, b):
                # 使用操作符"/"进行张量除法
                result_0 = a / b
                # 使用torch.div函数进行张量除法
                result_1 = torch.div(a, b)
                # 使用张量对象的div方法进行张量除法
                result_2 = a.div(b)

                # 返回三种不同方式的除法结果
                return result_0, result_1, result_2

        # 加载历史版本的模块
        try:
            # 使用特定路径加载模块，用于轻量级解释器
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_v2.ptl"
            )
        except Exception as e:
            # 加载失败则跳过当前测试
            self.skipTest("Failed to load fixture!")

        # 当前模块的移动版本
        current_mobile_module = self._save_load_mobile_module(MyModule)

        # 遍历输入样本的笛卡尔积
        for val_a, val_b in product(sample_input, sample_input):
            # 创建张量a和b，分别包含val_a和val_b的值
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            # 定义一个辅助函数，用于测试不同模块和函数的结果
            def _helper(m, fn):
                # 使用_try_fn方法获取模块m和函数fn在a和b上的结果
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                # 检查结果是否是异常
                if isinstance(m_results, Exception):
                    # 如果模块m的结果是异常，则函数fn的结果也应该是异常
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    # 否则逐一比较模块m的结果和函数fn的结果
                    for result in m_results:
                        self.assertEqual(result, fn_result)

            # 使用历史版本的除法函数测试v3_mobile_module和historic_div
            _helper(v3_mobile_module, historic_div)
            # 使用torch.div函数测试current_mobile_module
            _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # 总共生成10个样本示例
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # 生成一个整数和浮点数的组合样本
    @example((2, 3, 2.0, 3.0))  # 确保此示例会被覆盖测试
    # 定义一个测试方法，用于测试版本化的原位张量除法操作
    def test_versioned_div_tensor_inplace(self, sample_input):
        
        # 定义一个内部函数，用于历史版本的原位除法操作
        def historic_div_(self, other):
            # 如果操作数中有浮点数，则使用真除方法
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide_(other)
            # 否则使用截断模式进行整数除法
            return self.divide_(other, rounding_mode="trunc")

        # 定义一个简单的神经网络模块类，重载了 forward 方法
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                a /= b  # 原位除法操作
                return a
        
        try:
            # 尝试加载用于轻量级解释器的版本3模块
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_inplace_v2.ptl"
            )
        except Exception as e:
            # 加载失败时跳过测试，并提示加载失败
            self.skipTest("Failed to load fixture!")

        # 使用自定义方法保存和加载当前模块
        current_mobile_module = self._save_load_mobile_module(MyModule)

        # 对输入样本的笛卡尔积进行迭代测试
        for val_a, val_b in product(sample_input, sample_input):
            # 创建张量 a 和 b，其中元素值分别为 val_a 和 val_b
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            # 定义一个辅助函数，用于测试给定方法在两个模块上的结果
            def _helper(m, fn):
                # 尝试在模块 m 上应用函数 fn，并记录结果
                fn_result = self._try_fn(fn, a.clone(), b)
                m_result = self._try_fn(m, a, b)
                # 如果模块方法返回异常，断言函数 fn 也返回异常
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    # 否则断言模块方法和函数 fn 返回相同结果，并且结果等于张量 a
                    self.assertEqual(m_result, fn_result)
                    self.assertEqual(m_result, a)

            # 在版本3模块上使用历史版本的原位除法方法进行测试
            _helper(v3_mobile_module, historic_div_)

            # 由于 a 是原位修改过的，重新创建一个张量 a
            a = torch.tensor((val_a,))
            # 在当前模块上使用 Torch 提供的原位除法方法进行测试
            _helper(current_mobile_module, torch.Tensor.div_)

    @settings(
        max_examples=10, deadline=200000
    )  # 将生成总共10个示例
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # 生成一个整数和浮点数的对
    @example((2, 3, 2.0, 3.0))  # 确保此示例被覆盖测试
    # 定义一个测试方法，用于测试版本化的张量除法功能，接受一个样本输入作为参数
    def test_versioned_div_tensor_out(self, sample_input):
        
        # 定义一个内部方法，用于处理历史版本的除法操作，将结果写入指定的输出张量
        def historic_div_out(self, other, out):
            # 如果任何一个操作数是浮点数，或者输出张量是浮点数，则使用真除法计算结果
            if (
                self.is_floating_point()
                or other.is_floating_point()
                or out.is_floating_point()
            ):
                return torch.true_divide(self, other, out=out)
            # 否则，使用截断模式的除法计算结果
            return torch.divide(self, other, out=out, rounding_mode="trunc")

        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            # 前向传播方法，执行张量除法操作，将结果写入指定的输出张量
            def forward(self, a, b, out):
                return a.div(b, out=out)

        try:
            # 尝试加载用于轻量级解释器的模型，模型文件路径为指定的版本路径
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_out_v2.ptl"
            )
        except Exception as e:
            # 如果加载失败，则跳过此测试用例，输出失败信息
            self.skipTest("Failed to load fixture!")

        # 使用自定义方法 _save_load_mobile_module 加载当前模型的移动端版本
        current_mobile_module = self._save_load_mobile_module(MyModule)

        # 遍历样本输入的笛卡尔积
        for val_a, val_b in product(sample_input, sample_input):
            # 创建张量 a 和 b，分别初始化为 val_a 和 val_b
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            # 遍历不同类型的输出张量：一个空的浮点型张量和一个空的长整型张量
            for out in (torch.empty((1,)), torch.empty((1,), dtype=torch.long)):

                # 定义一个辅助方法 _helper，用于执行指定函数的结果比较
                def _helper(m, fn):
                    fn_result = None
                    # 如果函数是 torch.div，则使用复制后的输出张量执行函数
                    if fn is torch.div:
                        fn_result = self._try_fn(fn, a, b, out=out.clone())
                    else:
                        fn_result = self._try_fn(fn, a, b, out=out.clone())
                    # 使用指定模块执行函数，并将结果存储在 out 中
                    m_result = self._try_fn(m, a, b, out)

                    # 检查 m_result 的类型，如果是异常则验证 fn_result 也应该是异常
                    if isinstance(m_result, Exception):
                        self.assertTrue(fn_result, Exception)
                    else:
                        # 否则，验证 m_result 和 fn_result 相等，并且与 out 相等
                        self.assertEqual(m_result, fn_result)
                        self.assertEqual(m_result, out)

                # 分别使用历史版本模块和当前版本模块执行 _helper 方法
                _helper(v3_mobile_module, historic_div_out)
                _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # 总共生成 10 个例子
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # 生成一个整数和一个浮点数的元组作为输入样本
    @example((2, 3, 2.0, 3.0))  # 确保覆盖此示例
    # 定义测试方法，用于测试版本化的标量除法功能，接受一个样本输入参数
    def test_versioned_div_scalar(self, sample_input):
        
        # 定义一个处理历史浮点数标量除法的函数，参数是一个浮点数，返回除法结果
        def historic_div_scalar_float(self, other: float):
            return torch.true_divide(self, other)

        # 定义一个处理历史整数标量除法的函数，参数是一个整数，根据是否是浮点数执行不同的除法操作
        def historic_div_scalar_int(self, other: int):
            if self.is_floating_point():
                return torch.true_divide(self, other)
            return torch.divide(self, other, rounding_mode="trunc")

        # 定义一个继承自torch.nn.Module的类，实现了一个前向计算方法，实现浮点数除法操作
        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                return a / b

        # 定义一个继承自torch.nn.Module的类，实现了一个前向计算方法，实现整数除法操作
        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                return a / b

        try:
            # 尝试加载历史版本的浮点数除法模型
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/jit/fixtures/test_versioned_div_scalar_float_v2.ptl"
            )
            # 尝试加载历史版本的整数除法模型
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_int_v2.ptl"
            )
        except Exception as e:
            # 加载失败时跳过测试，并提示加载失败的原因
            self.skipTest("Failed to load fixture!")

        # 当前浮点数除法模型，保存和加载
        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        # 当前整数除法模型，保存和加载
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        # 使用给定的输入样本生成所有可能的组合
        for val_a, val_b in product(sample_input, sample_input):
            # 将输入值转换为张量a
            a = torch.tensor((val_a,))
            # 取当前的b值
            b = val_b

            # 定义一个辅助函数，接受一个模型和一个函数，测试它们在给定输入下的结果是否相同
            def _helper(m, fn):
                # 测试版本化模型m在输入a, b下的计算结果
                m_result = self._try_fn(m, a, b)
                # 测试标准函数fn在输入a, b下的计算结果
                fn_result = self._try_fn(fn, a, b)

                # 如果模型结果是异常，断言标准函数结果也应该是异常
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    # 否则，断言模型结果与标准函数结果应相等
                    self.assertEqual(m_result, fn_result)

            # 根据b的类型选择测试的模型和标准函数
            if isinstance(b, float):
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, historic_div_scalar_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # 总共生成10个样本
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # 生成一个整数和一个浮点数的对
    @example((2, 3, 2.0, 3.0))  # 确保覆盖这个例子
    # 定义测试方法：测试版本化的标量倒数除法功能，接受一个名为sample_input的参数
    def test_versioned_div_scalar_reciprocal(self, sample_input):
        # 定义一个内部方法：用于处理历史版本的浮点数标量倒数除法
        def historic_div_scalar_float_reciprocal(self, other: float):
            return other / self

        # 定义一个内部方法：用于处理历史版本的整数标量倒数除法
        def historic_div_scalar_int_reciprocal(self, other: int):
            # 如果当前对象是浮点数类型，则直接执行除法操作
            if self.is_floating_point():
                return other / self
            # 否则调用torch.divide进行整数除法操作，使用截断模式
            return torch.divide(other, self, rounding_mode="trunc")

        # 定义一个继承自torch.nn.Module的浮点数类型模块
        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                return b / a

        # 定义一个继承自torch.nn.Module的整数类型模块
        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                return b / a

        # 尝试加载预定义的Lite模块，如果失败则跳过测试
        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        # 使用自定义方法保存和加载当前的浮点数类型模块
        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        # 使用自定义方法保存和加载当前的整数类型模块
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        # 遍历生成的sample_input的笛卡尔积
        for val_a, val_b in product(sample_input, sample_input):
            # 创建torch张量a，包含值val_a
            a = torch.tensor((val_a,))
            # 设置b为val_b
            b = val_b

            # 定义一个辅助方法_helper，用于执行给定模块和函数的计算，并进行断言
            def _helper(m, fn):
                # 调用自定义方法_try_fn，计算模块m对输入a、b的计算结果
                m_result = self._try_fn(m, a, b)
                fn_result = None
                # 如果fn是torch.div，则反转参数顺序调用self._try_fn
                if fn is torch.div:
                    fn_result = self._try_fn(torch.div, b, a)
                else:
                    fn_result = self._try_fn(fn, a, b)

                # 如果m_result是异常，则断言fn_result也是异常
                if isinstance(m_result, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                # 否则如果fn是torch.div或a是浮点数，则断言m_result等于fn_result
                elif fn is torch.div or a.is_floating_point():
                    self.assertEqual(m_result, fn_result)
                # 否则跳过，因为fn不是torch.div且a是整数，historic_div_scalar_int执行截断除法
                else:
                    pass

            # 如果b是浮点数，执行以下操作
            if isinstance(b, float):
                # 对浮点数类型的Lite模块和当前浮点数类型模块执行_helper方法
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            # 否则执行以下操作
            else:
                # 对整数类型的Lite模块和当前整数类型模块执行_helper方法
                _helper(v3_mobile_module_int, current_mobile_module_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # 指定最大生成例子数为10，超时时间为200000毫秒
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # 指定sample_input的生成规则为生成包含整数和浮点数的元组
    @example((2, 3, 2.0, 3.0))  # 确保该例子会被覆盖到
    # 测试版本化的就地标量除法函数
    def test_versioned_div_scalar_inplace(self, sample_input):
        
        # 定义历史版本中的浮点数就地除法函数
        def historic_div_scalar_float_inplace(self, other: float):
            return self.true_divide_(other)

        # 定义历史版本中的整数就地除法函数
        def historic_div_scalar_int_inplace(self, other: int):
            # 如果张量是浮点型，使用真除法
            if self.is_floating_point():
                return self.true_divide_(other)
            # 否则使用截断模式进行除法运算
            return self.divide_(other, rounding_mode="trunc")

        # 定义用于浮点数的模块
        class MyModuleFloat(torch.nn.Module):
            def forward(self, a, b: float):
                a /= b
                return a

        # 定义用于整数的模块
        class MyModuleInt(torch.nn.Module):
            def forward(self, a, b: int):
                a /= b
                return a

        try:
            # 加载浮点数版本的移动端模块
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl"
            )
            # 加载整数版本的移动端模块
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl"
            )
        except Exception as e:
            # 如果加载失败，则跳过测试
            self.skipTest("Failed to load fixture!")

        # 使用当前浮点数模块进行保存加载测试
        current_mobile_module_float = self._save_load_module(MyModuleFloat)
        # 使用当前整数模块进行保存加载测试
        current_mobile_module_int = self._save_load_module(MyModuleInt)

        # 对样本输入的每一对值进行组合
        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            # 定义辅助函数，用于测试模块和张量函数的结果
            def _helper(m, fn):
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                # 如果模块返回异常，则断言函数也返回异常
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    # 否则断言模块和函数的结果相等
                    self.assertEqual(m_result, fn_result)

            # 根据 b 的类型选择适当的辅助函数进行测试
            if isinstance(b, float):
                _helper(current_mobile_module_float, torch.Tensor.div_)
            else:
                _helper(current_mobile_module_int, torch.Tensor.div_)

    # 注意：标量除法在操作版本 3 中已经是真除法，因此此测试验证其行为未更改。
    def test_versioned_div_scalar_scalar(self):
        # 定义一个测试用的 PyTorch 模块，计算四种除法结果
        class MyModule(torch.nn.Module):
            def forward(self, a: float, b: int, c: float, d: int):
                # 计算 a 除以 b 的结果
                result_0 = a / b
                # 计算 a 除以 c 的结果
                result_1 = a / c
                # 计算 b 除以 c 的结果
                result_2 = b / c
                # 计算 b 除以 d 的结果
                result_3 = b / d
                return (result_0, result_1, result_2, result_3)

        try:
            # 尝试加载预先准备好的 Lite 解释器用的 PyTorch 模块
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_scalar_v2.ptl"
            )
        except Exception as e:
            # 加载失败时跳过当前测试
            self.skipTest("Failed to load fixture!")

        # 使用自定义的 MyModule 模块进行序列化和反序列化测试
        current_mobile_module = self._save_load_mobile_module(MyModule)

        def _helper(m, fn):
            # 定义输入的测试值
            vals = (5.0, 3, 2.0, 7)
            # 分别使用两个模块计算结果
            m_result = m(*vals)
            fn_result = fn(*vals)
            # 对比两个模块的计算结果是否一致
            for mr, hr in zip(m_result, fn_result):
                self.assertEqual(mr, hr)

        # 调用 _helper 函数进行测试
        _helper(v3_mobile_module, current_mobile_module)

    def test_versioned_linspace(self):
        # 定义一个计算 torch.linspace 的 PyTorch 模块
        class Module(torch.nn.Module):
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                # 使用 torch.linspace 生成具有 5 个步长的张量
                c = torch.linspace(a, b, steps=5)
                # 使用 torch.linspace 生成具有 100 个步长的张量
                d = torch.linspace(a, b, steps=100)
                return c, d

        # 加载预先准备好的 Lite 解释器用的 PyTorch 模块
        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_v7.ptl"
        )

        # 将模型保存到字节流中，用于 Lite 解释器加载
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)

        # 使用自定义的 Module 模块进行序列化和反序列化测试
        current_mobile_module = self._save_load_mobile_module(Module)

        # 定义多组输入进行测试
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            # 使用 v7_mobile_module 计算输出
            (output_with_step, output_without_step) = v7_mobile_module(a, b)
            # 使用 current_mobile_module 计算输出
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # 验证当没有给定步长时，输出应该有 100 个元素
            self.assertTrue(output_without_step.size(dim=0) == 100)
            # 验证给定步长为 5 时，输出应该有 5 个元素
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # 验证输出应与最新版本的模块相等
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)
    def test_versioned_linspace_out(self):
        # 定义一个内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 forward 方法，用于模型的前向传播
            def forward(
                self,
                a: Union[int, float, complex],  # 参数 a 可以是 int、float 或 complex 类型
                b: Union[int, float, complex],  # 参数 b 可以是 int、float 或 complex 类型
                out: torch.Tensor,  # 参数 out 是一个 torch.Tensor，用于接收输出
            ):
                # 调用 torch.linspace 生成从 a 到 b 等间隔的 100 个数，将结果存入 out
                return torch.linspace(a, b, steps=100, out=out)

        # 指定模型路径
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_out_v7.ptl"
        )
        # 使用 torch.jit.load 加载模型
        loaded_model = torch.jit.load(model_path)
        # 将模型保存到字节流 buffer 中
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 使用 _load_for_lite_interpreter 函数加载模型为 v7_mobile_module
        v7_mobile_module = _load_for_lite_interpreter(buffer)
        # 调用 self._save_load_mobile_module 方法加载当前模型为 current_mobile_module
        current_mobile_module = self._save_load_mobile_module(Module)

        # 定义输入样本 sample_inputs
        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),  # 创建一个空的 int64 类型的 Tensor
                torch.empty((100,), dtype=torch.int64),  # 创建一个空的 int64 类型的 Tensor
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),  # 创建一个空的 int64 类型的 Tensor
                torch.empty((100,), dtype=torch.int64),  # 创建一个空的 int64 类型的 Tensor
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),  # 创建一个空的 float64 类型的 Tensor
                torch.empty((100,), dtype=torch.float64),  # 创建一个空的 float64 类型的 Tensor
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),  # 创建一个空的 complex64 类型的 Tensor
                torch.empty((100,), dtype=torch.complex64),  # 创建一个空的 complex64 类型的 Tensor
            ),
        )
        # 遍历输入样本，分别使用 v7_mobile_module 和 current_mobile_module 进行计算和比较
        for start, end, out_for_old, out_for_new in sample_inputs:
            # 使用 v7_mobile_module 进行计算，将结果存入 output
            output = v7_mobile_module(start, end, out_for_old)
            # 使用 current_mobile_module 进行计算，将结果存入 output_current
            output_current = current_mobile_module(start, end, out_for_new)
            # 断言输出的第一个维度大小为 100
            self.assertTrue(output.size(dim=0) == 100)
            # 断言输出结果应与 current_mobile_module 的结果相等
            self.assertEqual(output, output_current)
    # 定义一个测试方法，用于测试版本化的对数空间计算功能
    def test_versioned_logspace(self):
        # 定义一个内部类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义 Module 类的前向传播方法
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                # 使用 torch.logspace 生成指定范围内的对数空间张量 c，包含 5 个元素
                c = torch.logspace(a, b, steps=5)
                # 使用 torch.logspace 生成指定范围内的对数空间张量 d，包含 100 个元素
                d = torch.logspace(a, b, steps=100)
                # 返回生成的两个对数空间张量
                return c, d

        # 从文件加载经过脚本化的模块
        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_v8.ptl"
        )

        # 将模型保存到 BytesIO 缓冲区
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 使用自定义函数 _load_for_lite_interpreter 加载为 Lite 解释器下的模块
        v8_mobile_module = _load_for_lite_interpreter(buffer)

        # 调用 self._save_load_mobile_module 方法加载当前的移动端模块
        current_mobile_module = self._save_load_mobile_module(Module)

        # 定义输入样本集合
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        # 遍历输入样本集合
        for a, b in sample_inputs:
            # 使用 v8_mobile_module 对输入 a, b 进行前向传播，获取输出张量
            (output_with_step, output_without_step) = v8_mobile_module(a, b)
            # 使用 current_mobile_module 对输入 a, b 进行前向传播，获取输出张量
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # 断言未给定步长时，output_without_step 张量的大小应为 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            # 断言给定步长时，output_with_step 张量的大小应为 5
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # 断言 output_with_step 与 current_with_step 张量相等
            self.assertEqual(output_with_step, current_with_step)
            # 断言 output_without_step 与 current_without_step 张量相等
            self.assertEqual(output_without_step, current_without_step)
    def test_versioned_logspace_out(self):
        # 定义一个测试方法，用于测试版本化的 logspace 输出功能

        class Module(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的内部类 Module

            def forward(
                self,
                a: Union[int, float, complex],
                b: Union[int, float, complex],
                out: torch.Tensor,
            ):
                # 定义 Module 类的前向传播方法，接受两个数值和一个输出张量作为参数
                return torch.logspace(a, b, steps=100, out=out)
                # 返回 torch.logspace 的结果，生成从 10^a 到 10^b 的步数为 100 的对数空间张量

        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_out_v8.ptl"
        )
        # 设置模型文件路径

        loaded_model = torch.jit.load(model_path)
        # 加载模型

        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        # 将模型保存为字节流

        buffer.seek(0)
        # 将字节流的读取指针移到开头

        v8_mobile_module = _load_for_lite_interpreter(buffer)
        # 使用特定的函数加载 lite 解释器中的模型版本

        current_mobile_module = self._save_load_mobile_module(Module)
        # 使用自定义函数加载当前移动模块的 Module 类

        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),
                torch.empty((100,), dtype=torch.float64),
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),
                torch.empty((100,), dtype=torch.complex64),
            ),
        )
        # 定义多个输入样本，每个样本包含起始点、终止点和两个输出张量

        for start, end, out_for_old, out_for_new in sample_inputs:
            # 遍历样本输入

            output = v8_mobile_module(start, end, out_for_old)
            # 使用 v8_mobile_module 进行推断，生成输出张量

            output_current = current_mobile_module(start, end, out_for_new)
            # 使用 current_mobile_module 进行推断，生成输出张量

            # 断言：当没有指定步数时，输出张量的大小应为 100
            self.assertTrue(output.size(dim=0) == 100)

            # 断言：升级后的模型输出应与新版本输出相匹配
            self.assertEqual(output, output_current)
```