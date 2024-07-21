# `.\pytorch\test\jit\test_upgraders.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的库
import io
import os
import sys
import zipfile
from typing import Union

# 导入 PyTorch 相关模块
import torch
from torch.testing import FileCheck

# 将测试文件夹 test/ 添加到系统路径，以便导入其它模块
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入测试框架中的 JIT 测试用例基类
from torch.testing._internal.jit_utils import JitTestCase

# 如果此文件被直接运行，抛出运行时错误提醒用户正确运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestUpgraders，继承自 JitTestCase
class TestUpgraders(JitTestCase):

    # 辅助函数：加载模型的版本信息
    def _load_model_version(self, loaded_model):
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将加载的模型保存到字节流缓冲区中
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        # 使用字节流创建 ZipFile 对象
        zipped_model = zipfile.ZipFile(buffer)

        # 尝试读取存储在 Zip 文件中的版本号信息
        try:
            version = int(zipped_model.read("archive/version").decode("utf-8"))
        except KeyError:
            # 如果指定文件不存在，则读取另一种路径下的版本号信息
            version = int(zipped_model.read("archive/.data/version").decode("utf-8"))
        
        # 返回读取到的版本号
        return version

    # 测试函数：测试已填充的升级器图
    # TODO (tugsuu) 我们应该理想情况下生成这些测试用例。
    def test_populated_upgrader_graph(self):
        # 定义一个通过 JIT 脚本化的简单函数 f
        @torch.jit.script
        def f():
            return 0
        
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将函数 f 保存到字节流缓冲区中
        torch.jit.save(f, buffer)
        buffer.seek(0)
        
        # 从字节流缓冲区中加载模型
        torch.jit.load(buffer)
        
        # 获取当前升级器映射的大小和内容的 dump
        upgraders_size = torch._C._get_upgraders_map_size()
        upgraders_dump = torch._C._dump_upgraders_map()
        
        # 确保我们只填充了一次升级器映射，
        # 然后再次加载并确保升级器映射内容相同
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size_second_time = torch._C._get_upgraders_map_size()
        upgraders_dump_second_time = torch._C._dump_upgraders_map()
        
        # 断言两次升级器映射的大小和内容一致
        self.assertTrue(upgraders_size == upgraders_size_second_time)
        self.assertTrue(upgraders_dump == upgraders_dump_second_time)
    # 测试函数：向操作符版本映射中添加新条目并进行验证
    def test_add_value_to_version_map(self):
        # 获取测试前的操作符版本映射
        map_before_test = torch._C._get_operator_version_map()

        # 设置一个升级后的版本号
        upgrader_bumped_version = 3
        # 设置升级器的名称
        upgrader_name = "_test_serialization_subcmul_0_2"
        # 设置升级器的模式字符串
        upgrader_schema = "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"
        # 创建一个升级器条目对象
        dummy_entry = torch._C._UpgraderEntry(
            upgrader_bumped_version, upgrader_name, upgrader_schema
        )

        # 向操作符版本映射中添加测试用的条目
        torch._C._test_only_add_entry_to_op_version_map(
            "aten::_test_serialization_subcmul", dummy_entry
        )
        # 获取测试后的操作符版本映射
        map_after_test = torch._C._get_operator_version_map()

        # 断言：确保添加的操作符在映射中
        self.assertTrue("aten::_test_serialization_subcmul" in map_after_test)
        # 断言：确保映射的大小增加了一个条目
        self.assertTrue(len(map_after_test) - len(map_before_test) == 1)

        # 从操作符版本映射中移除测试用的条目
        torch._C._test_only_remove_entry_to_op_version_map(
            "aten::_test_serialization_subcmul"
        )
        # 获取移除测试后的操作符版本映射
        map_after_remove_test = torch._C._get_operator_version_map()

        # 断言：确保移除的操作符不再在映射中
        self.assertTrue(
            "aten::_test_serialization_subcmul" not in map_after_remove_test
        )
        # 断言：确保移除后映射的大小与测试前相同
        self.assertEqual(len(map_after_remove_test), len(map_before_test))

    # 测试函数：测试填充升级器图
    def test_populated_test_upgrader_graph(self):
        # 使用 Torch 的 JIT 脚本定义一个简单函数 f
        @torch.jit.script
        def f():
            return 0

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将函数 f 保存到缓冲区中
        torch.jit.save(f, buffer)
        buffer.seek(0)
        # 从缓冲区加载函数 f
        torch.jit.load(buffer)

        # 获取填充后的升级器图的大小
        upgraders_size = torch._C._get_upgraders_map_size()

        # 构建一个测试用的映射
        test_map = {"a": str(torch._C.Graph()), "c": str(torch._C.Graph())}
        # 仅测试目的：填充升级器图
        torch._C._test_only_populate_upgraders(test_map)
        # 获取填充后的升级器图的新大小
        upgraders_size_after_test = torch._C._get_upgraders_map_size()

        # 断言：确保填充后升级器图大小增加了两个条目
        self.assertEqual(upgraders_size_after_test - upgraders_size, 2)

        # 获取升级器图的转储内容
        upgraders_dump = torch._C._dump_upgraders_map()
        # 断言：确保 'a' 和 'c' 在升级器图的转储内容中
        self.assertTrue("a" in upgraders_dump)
        self.assertTrue("c" in upgraders_dump)

        # 仅测试目的：移除填充的升级器图内容
        torch._C._test_only_remove_upgraders(test_map)
        # 获取移除填充后的升级器图的大小
        upgraders_size_after_remove_test = torch._C._get_upgraders_map_size()

        # 断言：确保移除填充后升级器图的大小与原始大小相同
        self.assertTrue(upgraders_size_after_remove_test == upgraders_size)

        # 获取移除填充后的升级器图的转储内容
        upgraders_dump_after_remove_test = torch._C._dump_upgraders_map()
        # 断言：确保 'a' 和 'c' 不再在移除填充后的升级器图的转储内容中
        self.assertTrue("a" not in upgraders_dump_after_remove_test)
        self.assertTrue("c" not in upgraders_dump_after_remove_test)
    def test_aten_div_tensor_at_3(self):
        # 设置模型路径
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_v3.pt"
        # 加载模型
        loaded_model = torch.jit.load(model_path)
        
        # 在模型中有 3 个 aten::div 操作
        # upgrader for aten::div 使用两个 div，因为存在 if/else 分支
        # 检查模型中是否存在 prim::If 节点
        FileCheck().check("prim::If").run(loaded_model.graph)
        # 检查 aten::div 操作的数量是否为 6
        FileCheck().check_count("aten::div", 6).run(loaded_model.graph)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将加载的模型保存到字节流缓冲区
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        
        # 加载模型版本
        version = self._load_model_version(loaded_model)
        # 断言模型版本为 4
        self.assertTrue(version == 4)
        
        # 再次从字节流缓冲区加载模型
        loaded_model_twice = torch.jit.load(buffer)
        # 检查模型的代码是否相等，因为图变量名可能每次不同
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    def test_aten_full_other_variants(self):
        def test_func():
            # 创建一个形状为 [4, 5, 6]，填充为 4 的张量，指定维度名称和数据类型
            a = torch.full([4, 5, 6], 4, names=["a", "b", "c"], dtype=torch.int64)
            return a
        
        # 对 test_func 进行脚本化
        scripted_func = torch.jit.script(test_func)
        buffer = io.BytesIO()
        # 将脚本化的函数保存到字节流缓冲区
        torch.jit.save(scripted_func, buffer)

        # 获取当前版本计算器标志的值
        current_flag_value = torch._C._get_version_calculator_flag()
        # 基于旧版本计算包版本
        torch._C._calculate_package_version_based_on_upgraders(False)
        buffer.seek(0)
        # 从字节流缓冲区加载函数
        loaded_func = torch.jit.load(buffer)
        # 加载函数的版本
        version = self._load_model_version(loaded_func)
        # 断言函数版本为 5
        self.assertTrue(version == 5)

        # 基于新版本计算包版本
        torch._C._calculate_package_version_based_on_upgraders(True)
        buffer.seek(0)
        # 再次从字节流缓冲区加载函数
        loaded_func = torch.jit.load(buffer)
        # 加载函数的版本
        version = self._load_model_version(loaded_func)
        # 断言函数版本为 5
        self.assertTrue(version == 5)

        # 确保保留旧的行为
        torch._C._calculate_package_version_based_on_upgraders(current_flag_value)

    def test_aten_linspace(self):
        # 设置模型路径
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_v7.ptl"
        # 加载模型
        loaded_model = torch.jit.load(model_path)
        # 定义样本输入
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            # 使用 loaded_model 进行计算
            output_with_step, output_without_step = loaded_model(a, b)
            # 当没有给定步长时，应该使用 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            # 确保有步长时的输出大小为 5
            self.assertTrue(output_with_step.size(dim=0) == 5)

        # 加载模型版本
        version = self._load_model_version(loaded_model)
        # 断言模型版本为 8
        self.assertTrue(version == 8)
    # 定义测试函数 test_aten_linspace_out，用于测试加载的 TorchScript 模型对于 torch.linspace 函数的输出是否正确
    def test_aten_linspace_out(self):
        # 设置模型路径为特定版本的测试文件
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_out_v7.ptl"
        )
        # 加载 TorchScript 模型
        loaded_model = torch.jit.load(model_path)
        # 设置多组输入参数进行测试
        sample_inputs = (
            (3, 10, torch.empty((100,), dtype=torch.int64)),
            (-10, 10, torch.empty((100,), dtype=torch.int64)),
            (4.0, 6.0, torch.empty((100,), dtype=torch.float64)),
            (3 + 4j, 4 + 5j, torch.empty((100,), dtype=torch.complex64)),
        )
        # 遍历每组输入参数
        for a, b, c in sample_inputs:
            # 调用加载的模型执行计算
            output = loaded_model(a, b, c)
            # 断言输出的第一维度大小为 100
            self.assertTrue(output.size(dim=0) == 100)

        # 获取模型的版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本为 8
        self.assertTrue(version == 8)

    # 定义测试函数 test_aten_logspace，用于测试加载的 TorchScript 模型对于 torch.logspace 函数的输出是否正确
    def test_aten_logspace(self):
        # 设置模型路径为特定版本的测试文件
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_v8.ptl"
        # 加载 TorchScript 模型
        loaded_model = torch.jit.load(model_path)
        # 设置多组输入参数进行测试
        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        # 遍历每组输入参数
        for a, b in sample_inputs:
            # 调用加载的模型执行计算
            output_with_step, output_without_step = loaded_model(a, b)
            # 断言输出的第一维度大小为 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            # 断言输出的第一维度大小为 5
            self.assertTrue(output_with_step.size(dim=0) == 5)

        # 获取模型的版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本为 9
        self.assertTrue(version == 9)

    # 定义测试函数 test_aten_logspace_out，用于测试加载的 TorchScript 模型对于 torch.logspace 函数的输出是否正确
    def test_aten_logspace_out(self):
        # 设置模型路径为特定版本的测试文件
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_out_v8.ptl"
        )
        # 加载 TorchScript 模型
        loaded_model = torch.jit.load(model_path)
        # 设置多组输入参数进行测试
        sample_inputs = (
            (3, 10, torch.empty((100,), dtype=torch.int64)),
            (-10, 10, torch.empty((100,), dtype=torch.int64)),
            (4.0, 6.0, torch.empty((100,), dtype=torch.float64)),
            (3 + 4j, 4 + 5j, torch.empty((100,), dtype=torch.complex64)),
        )
        # 遍历每组输入参数
        for a, b, c in sample_inputs:
            # 调用加载的模型执行计算
            output = loaded_model(a, b, c)
            # 断言输出的第一维度大小为 100
            self.assertTrue(output.size(dim=0) == 100)

        # 获取模型的版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本为 9
        self.assertTrue(version == 9)
    def test_aten_test_serialization(self):
        model_path = (
            pytorch_test_dir + "/jit/fixtures/_test_serialization_subcmul_v2.pt"
        )

        # add test version entry to the version map
        upgrader_bumped_version = 3  # 定义增加的版本号为3
        upgrader_name = "_test_serialization_subcmul_0_2"  # 定义升级器的名称
        upgrader_schema = "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"  # 定义升级器的函数签名
        dummy_entry = torch._C._UpgraderEntry(
            upgrader_bumped_version, upgrader_name, upgrader_schema
        )

        torch._C._test_only_add_entry_to_op_version_map(
            "aten::_test_serialization_subcmul", dummy_entry
        )

        # add test upgrader in the upgraders map
        @torch.jit.script
        def _test_serialization_subcmul_0_2(
            self: torch.Tensor, other: torch.Tensor, alpha: Union[int, float] = 2
        ) -> torch.Tensor:
            return other - (self * alpha)

        torch._C._test_only_populate_upgraders(
            {
                "_test_serialization_subcmul_0_2": str(
                    _test_serialization_subcmul_0_2.graph
                )
            }
        )

        # test if the server is able to find the test upgraders and apply to IR
        loaded_model = torch.jit.load(model_path)  # 加载模型
        FileCheck().check_count("aten::mul", 2).run(loaded_model.graph)  # 检查加载的模型中乘法操作的数量
        FileCheck().check_count("aten::sub", 2).run(loaded_model.graph)  # 检查加载的模型中减法操作的数量

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)  # 将加载的模型保存到字节流中
        buffer.seek(0)
        version = self._load_model_version(loaded_model)  # 获取加载模型的版本信息
        self.assertTrue(version == 3)  # 断言加载模型的版本为3
        loaded_model_twice = torch.jit.load(buffer)  # 从保存的字节流中再次加载模型

        # we check by its' code because graph variable names
        # can be different every time
        self.assertEqual(loaded_model.code, loaded_model_twice.code)  # 检查两次加载的模型代码是否相同

        torch._C._test_only_remove_entry_to_op_version_map(
            "aten::_test_serialization_subcmul"
        )

        torch._C._test_only_remove_upgraders(
            {
                "_test_serialization_subcmul_0_2": str(
                    _test_serialization_subcmul_0_2.graph
                )
            }
        )
    # 测试函数，用于验证在特定模型中的运算和操作
    def test_aten_div_tensor_out_at_3(self):
        # 设置模型文件路径
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_out_v3.pt"
        )
        # 加载模型
        loaded_model = torch.jit.load(model_path)
        # 检查加载后的模型图中是否包含 "prim::If"
        FileCheck().check("prim::If").run(loaded_model.graph)
        # 检查加载后的模型图中 "aten::div" 出现的次数是否为 2
        FileCheck().check_count("aten::div", 2).run(loaded_model.graph)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将加载的模型保存到字节流中
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        # 加载保存在字节流中的模型，并获取其版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本号为 4
        self.assertTrue(version == 4)
        # 第二次加载保存在字节流中的模型
        loaded_model_twice = torch.jit.load(buffer)
        # 通过比较模型的代码内容来验证模型是否相同，因为模型图中的变量名可能不同
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    # 测试函数，用于验证在特定模型中的运算和操作
    def test_aten_full_at_4(self):
        # 设置模型文件路径
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_full_integer_value_v4.pt"
        )
        # 加载模型
        loaded_model = torch.jit.load(model_path)
        # 检查加载后的模型图中 "aten::Float" 出现的次数是否为 1
        FileCheck().check_count("aten::Float", 1).run(loaded_model.graph)
        # 检查加载后的模型图中 "aten::full" 出现的次数是否为 2
        FileCheck().check_count("aten::full", 2).run(loaded_model.graph)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将加载的模型保存到字节流中
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        # 加载保存在字节流中的模型，并获取其版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本号为 5
        self.assertTrue(version == 5)
        # 第二次加载保存在字节流中的模型
        loaded_model_twice = torch.jit.load(buffer)
        # 通过比较模型的代码内容来验证模型是否相同，因为模型图中的变量名可能不同
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    # 测试函数，用于验证在特定模型中的运算和操作
    def test_aten_full_out_at_4(self):
        # 设置模型文件路径
        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_full_preserved_v4.pt"
        )
        # 加载模型
        loaded_model = torch.jit.load(model_path)
        # 检查加载后的模型图中 "aten::full" 出现的次数是否为 5
        FileCheck().check_count("aten::full", 5).run(loaded_model.graph)
        # 获取模型的版本信息
        version = self._load_model_version(loaded_model)
        # 断言模型的版本号为 5
        self.assertTrue(version == 5)
```