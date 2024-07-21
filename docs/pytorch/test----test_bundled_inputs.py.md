# `.\pytorch\test\test_bundled_inputs.py`

```py
# 指定脚本解释器为 Python 3
#!/usr/bin/env python3
# 指定所有者为移动端的责任人
# mypy: allow-untyped-defs

# 导入必要的模块
import io  # 导入 io 模块，用于处理字节流
import textwrap  # 导入 textwrap 模块，用于格式化文本
from typing import Dict, List, Optional  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 模块
import torch.utils.bundled_inputs  # 导入 PyTorch 的 bundled_inputs 模块
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入 PyTorch 测试相关的模块


# 定义函数：计算模型在内存中的大小（字节数）
def model_size(sm):
    buffer = io.BytesIO()  # 创建一个字节流对象
    torch.jit.save(sm, buffer)  # 将模型 sm 保存到字节流中
    return len(buffer.getvalue())  # 返回字节流中内容的长度（即模型的大小）


# 定义函数：保存并加载模型
def save_and_load(sm):
    buffer = io.BytesIO()  # 创建一个字节流对象
    torch.jit.save(sm, buffer)  # 将模型 sm 保存到字节流中
    buffer.seek(0)  # 将字节流指针移动到开头
    return torch.jit.load(buffer)  # 从字节流中加载模型并返回


# 定义测试类 TestBundledInputs，继承自 TestCase 类
class TestBundledInputs(TestCase):
    # 定义单个张量模型的测试方法
    def test_single_tensors(self):
        # 定义一个简单的 Torch 模型，其 forward 方法直接返回输入参数
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # 使用 Torch 的 JIT 编译功能将模型编译为 TorchScript
        sm = torch.jit.script(SingleTensorModel())
        # 记录原始模型的大小
        original_size = model_size(sm)
        # 初始化一个空列表，用于存储获取表达式
        get_expr: List[str] = []
        # 定义多个示例张量
        samples = [
            # 小 numel 和小存储的张量
            (torch.tensor([1]),),
            # 大 numel 和小存储的张量
            (torch.tensor([[2, 3, 4]]).expand(1 << 16, -1)[:, ::2],),
            # 小 numel 和大存储的张量
            (torch.tensor(range(1 << 16))[-8:],),
            # 大零张量
            (torch.zeros(1 << 16),),
            # 大 channels-last 的全 1 张量
            (torch.ones(4, 8, 32, 32).contiguous(memory_format=torch.channels_last),),
            # 特殊编码的随机张量
            (torch.utils.bundled_inputs.bundle_randn(1 << 16),),
            # 量化的均匀张量
            (torch.quantize_per_tensor(torch.zeros(4, 8, 32, 32), 1, 0, torch.qint8),),
        ]
        # 使用 Torch 的 bundled_inputs 工具函数扩展模型以包含示例张量
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm, samples, get_expr
        )
        # 确保获取表达式列表中有内容
        # print(get_expr[0])
        # print(sm._generate_bundled_inputs.code)

        # 确保尽管有名义上大的捆绑输入，模型增长不多
        augmented_size = model_size(sm)
        self.assertLess(augmented_size, original_size + (1 << 12))

        # 将模型保存并加载
        loaded = save_and_load(sm)
        # 获取加载后的所有捆绑输入
        inflated = loaded.get_all_bundled_inputs()
        # 断言加载后的捆绑输入数量与示例张量数量相等
        self.assertEqual(loaded.get_num_bundled_inputs(), len(samples))
        # 断言加载后的捆绑输入列表长度与示例张量列表长度相等
        self.assertEqual(len(inflated), len(samples))
        # 断言加载后的第一个捆绑输入与其本身相等
        self.assertTrue(loaded(*inflated[0]) is inflated[0][0])

        # 遍历加载后的每个捆绑输入
        for idx, inp in enumerate(inflated):
            # 断言每个捆绑输入是一个元组
            self.assertIsInstance(inp, tuple)
            # 断言每个捆绑输入只包含一个张量
            self.assertEqual(len(inp), 1)
            # 断言每个捆绑输入的张量与对应示例张量在 dtype 上相等
            self.assertIsInstance(inp[0], torch.Tensor)
            if idx != 5:
                # 对于非第 6 个示例，检查张量的步长以便进行基准测试
                self.assertEqual(inp[0].stride(), samples[idx][0].stride())
                self.assertEqual(inp[0], samples[idx][0], exact_dtype=True)

        # 对于第 6 个示例，这个张量是随机生成的，通过大量试验验证其均值和标准差的范围
        self.assertEqual(inflated[5][0].shape, (1 << 16,))
        self.assertEqual(inflated[5][0].mean().item(), 0, atol=0.025, rtol=0)
        self.assertEqual(inflated[5][0].std().item(), 1, atol=0.02, rtol=0)
    def test_large_tensor_with_inflation(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # 使用 torch.jit.script 将 SingleTensorModel 脚本化
        sm = torch.jit.script(SingleTensorModel())
        # 创建一个大小为 2^16 的随机张量
        sample_tensor = torch.randn(1 << 16)
        # 将张量使用自定义膨胀函数打包，即使大小很大也可以处理，即使膨胀函数是恒等映射
        sample = torch.utils.bundled_inputs.bundle_large_tensor(sample_tensor)
        # 将打包后的输入与模型结合起来
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, [(sample,)])

        # 保存和加载模型
        loaded = save_and_load(sm)
        # 获取加载后模型的所有打包输入
        inflated = loaded.get_all_bundled_inputs()
        # 断言只有一个打包输入
        self.assertEqual(len(inflated), 1)
        # 断言加载后的第一个输入与原始样本张量相等
        self.assertEqual(inflated[0][0], sample_tensor)

    def test_rejected_tensors(self):
        def check_tensor(sample):
            # 在此作用域内定义类，以便每次运行都获得新类型
            class SingleTensorModel(torch.nn.Module):
                def forward(self, arg):
                    return arg

            sm = torch.jit.script(SingleTensorModel())
            # 断言会抛出异常并且异常消息包含 "Bundled input argument"
            with self.assertRaisesRegex(Exception, "Bundled input argument"):
                torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                    sm, [(sample,)]
                )

        # 普通的大张量
        check_tensor(torch.randn(1 << 16))
        # 这个张量有两个元素，但它们在内存中相距很远。
        # 目前我们无法紧凑地表示这种张量同时保持其步幅。
        small_sparse = torch.randn(2, 1 << 16)[:, 0:1]
        self.assertEqual(small_sparse.numel(), 2)
        # 检查小稀疏张量是否被拒绝
        check_tensor(small_sparse)

    def test_non_tensors(self):
        class StringAndIntModel(torch.nn.Module):
            def forward(self, fmt: str, num: int):
                return fmt.format(num)

        sm = torch.jit.script(StringAndIntModel())
        # 样本列表，包含字符串格式和整数
        samples = [
            ("first {}", 1),
            ("second {}", 2),
        ]
        # 将样本列表作为输入与模型结合
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, samples)

        # 保存和加载模型
        loaded = save_and_load(sm)
        # 获取加载后模型的所有打包输入
        inflated = loaded.get_all_bundled_inputs()
        # 断言加载后的打包输入与原始样本列表相等
        self.assertEqual(inflated, samples)
        # 断言加载后模型的第一个打包输入能正确输出 "first 1"
        self.assertTrue(loaded(*inflated[0]) == "first 1")
    def test_multiple_methods_with_inputs(self):
        class MultipleMethodModel(torch.nn.Module):
            def forward(self, arg):
                return arg

            @torch.jit.export
            def foo(self, arg):
                return arg

        # 将 MultipleMethodModel 转换为 TorchScript 模型
        mm = torch.jit.script(MultipleMethodModel())

        # 不同类型和形状的张量样本
        samples = [
            # Tensor with small numel and small storage.
            (torch.tensor([1]),),
            # Tensor with large numel and small storage.
            (torch.tensor([[2, 3, 4]]).expand(1 << 16, -1)[:, ::2],),
            # Tensor with small numel and large storage.
            (torch.tensor(range(1 << 16))[-8:],),
            # Large zero tensor.
            (torch.zeros(1 << 16),),
            # Large channels-last ones tensor.
            (torch.ones(4, 8, 32, 32).contiguous(memory_format=torch.channels_last),),
        ]

        # 每个样本的描述信息
        info = [
            "Tensor with small numel and small storage.",
            "Tensor with large numel and small storage.",
            "Tensor with small numel and large storage.",
            "Large zero tensor.",
            "Large channels-last ones tensor.",
            "Special encoding of random tensor.",
        ]

        # 使用 bundled_inputs 工具函数增强模型的多个方法
        torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
            mm,
            inputs={mm.forward: samples, mm.foo: samples},
            info={mm.forward: info, mm.foo: info},
        )

        # 将增强后的模型保存并加载
        loaded = save_and_load(mm)
        # 获取加载后模型的所有捆绑输入
        inflated = loaded.get_all_bundled_inputs()

        # 确保不同函数的捆绑输入是一致的
        self.assertEqual(inflated, samples)
        self.assertEqual(inflated, loaded.get_all_bundled_inputs_for_forward())
        self.assertEqual(inflated, loaded.get_all_bundled_inputs_for_foo())

        # 检查运行和大小辅助函数
        self.assertTrue(loaded(*inflated[0]) is inflated[0][0])
        self.assertEqual(loaded.get_num_bundled_inputs(), len(samples))

        # 检查适用于所有函数的辅助函数
        all_info = loaded.get_bundled_inputs_functions_and_info()
        self.assertEqual(set(all_info.keys()), {"forward", "foo"})
        self.assertEqual(
            all_info["forward"]["get_inputs_function_name"],
            ["get_all_bundled_inputs_for_forward"],
        )
        self.assertEqual(
            all_info["foo"]["get_inputs_function_name"],
            ["get_all_bundled_inputs_for_foo"],
        )
        self.assertEqual(all_info["forward"]["info"], info)
        self.assertEqual(all_info["foo"]["info"], info)

        # 示例展示如何将 'get_inputs_function_name' 转换为实际的捆绑输入列表
        for func_name in all_info.keys():
            input_func_name = all_info[func_name]["get_inputs_function_name"][0]
            func_to_run = getattr(loaded, input_func_name)
            self.assertEqual(func_to_run(), samples)
    def test_multiple_methods_with_inputs_both_defined_failure(self):
        class MultipleMethodModel(torch.nn.Module):
            def forward(self, arg):
                return arg

            @torch.jit.export
            def foo(self, arg):
                return arg

        samples = [(torch.tensor([1]),)]

        # inputs defined 2 ways so should fail
        # 断言捕获异常，测试多个方法同时定义输入的失败情况
        with self.assertRaises(Exception):
            mm = torch.jit.script(MultipleMethodModel())
            # 定义一个方法，用于生成前向函数的捆绑输入
            definition = textwrap.dedent(
                """
                def _generate_bundled_inputs_for_forward(self):
                    return []
                """
            )
            mm.define(definition)
            # 使用捆绑输入增强多个模型函数
            torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
                mm,
                inputs={
                    mm.forward: samples,
                    mm.foo: samples,
                },
            )

    def test_multiple_methods_with_inputs_neither_defined_failure(self):
        class MultipleMethodModel(torch.nn.Module):
            def forward(self, arg):
                return arg

            @torch.jit.export
            def foo(self, arg):
                return arg

        samples = [(torch.tensor([1]),)]

        # inputs not defined so should fail
        # 断言捕获异常，测试多个方法同时未定义输入的失败情况
        with self.assertRaises(Exception):
            mm = torch.jit.script(MultipleMethodModel())
            # 调用一个方法来生成前向函数的捆绑输入
            mm._generate_bundled_inputs_for_forward()
            # 使用捆绑输入增强多个模型函数
            torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
                mm,
                inputs={
                    mm.forward: None,
                    mm.foo: samples,
                },
            )

    def test_bad_inputs(self):
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # Non list for input list
        # 断言捕获异常，测试非列表输入的情况
        with self.assertRaises(TypeError):
            m = torch.jit.script(SingleTensorModel())
            # 使用捆绑输入增强模型函数
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m, inputs="foo"  # type: ignore[arg-type]
            )

        # List of non tuples. Most common error using the api.
        # 断言捕获异常，测试包含非元组的列表输入的情况
        with self.assertRaises(TypeError):
            m = torch.jit.script(SingleTensorModel())
            # 使用捆绑输入增强模型函数
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m,
                inputs=[
                    torch.ones(1, 2),  # type: ignore[list-item]
                ],
            )
    # 定义一个测试方法，用于测试双重增强失败的情况
    def test_double_augment_fail(self):
        # 定义一个简单的神经网络模型，只是简单地将输入参数返回
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # 使用 Torch 的 JIT 脚本化功能将模型 m 实例化为脚本化模型
        m = torch.jit.script(SingleTensorModel())
        # 使用 bundled_inputs 工具为模型 m 增强输入
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            m, inputs=[(torch.ones(1),)]
        )
        # 使用 assertRaisesRegex 断言上下文管理器来检查是否会抛出特定的异常
        with self.assertRaisesRegex(
            Exception, "Models can only be augmented with bundled inputs once."
        ):
            # 再次使用 bundled_inputs 工具增强模型 m，预期会抛出异常
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m, inputs=[(torch.ones(1),)]
            )

    # 定义一个测试方法，用于测试双重增强但非变异的情况
    def test_double_augment_non_mutator(self):
        # 定义一个简单的神经网络模型，只是简单地将输入参数返回
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # 使用 Torch 的 JIT 脚本化功能将模型 m 实例化为脚本化模型
        m = torch.jit.script(SingleTensorModel())
        # 使用 bundled_inputs 工具将模型 m 与输入参数捆绑在一起
        bundled_model = torch.utils.bundled_inputs.bundle_inputs(
            m, inputs=[(torch.ones(1),)]
        )
        # 使用 assertRaises 断言上下文管理器来检查是否会抛出特定的异常类型
        with self.assertRaises(AttributeError):
            # 检查原始模型 m 是否具有获取所有捆绑输入的方法，预期会抛出 AttributeError
            m.get_all_bundled_inputs()
        # 使用 assertEqual 断言检查捆绑后的模型 bundled_model 的输入是否与预期相符
        self.assertEqual(bundled_model.get_all_bundled_inputs(), [(torch.ones(1),)])
        # 使用 assertEqual 断言检查捆绑后的模型 bundled_model 的前向传播结果是否与预期相符
        self.assertEqual(bundled_model.forward(torch.ones(1)), torch.ones(1))

    # 定义一个测试方法，用于测试成功进行双重增强的情况
    def test_double_augment_success(self):
        # 定义一个简单的神经网络模型，只是简单地将输入参数返回
        class SingleTensorModel(torch.nn.Module):
            def forward(self, arg):
                return arg

        # 使用 Torch 的 JIT 脚本化功能将模型 m 实例化为脚本化模型
        m = torch.jit.script(SingleTensorModel())
        # 使用 bundled_inputs 工具将模型 m 与输入参数捆绑在一起
        bundled_model = torch.utils.bundled_inputs.bundle_inputs(
            m, inputs={m.forward: [(torch.ones(1),)]}
        )
        # 使用 assertEqual 断言检查捆绑后的模型 bundled_model 的输入是否与预期相符
        self.assertEqual(bundled_model.get_all_bundled_inputs(), [(torch.ones(1),)])

        # 使用 bundled_inputs 工具再次增强捆绑模型 bundled_model
        bundled_model2 = torch.utils.bundled_inputs.bundle_inputs(
            bundled_model, inputs=[(torch.ones(2),)]
        )
        # 使用 assertEqual 断言检查再次增强后的捆绑模型 bundled_model2 的输入是否与预期相符
        self.assertEqual(bundled_model2.get_all_bundled_inputs(), [(torch.ones(2),)])
# 如果当前脚本作为主程序执行（而不是被导入作为模块），则运行 `run_tests()` 函数。
if __name__ == "__main__":
    run_tests()
```