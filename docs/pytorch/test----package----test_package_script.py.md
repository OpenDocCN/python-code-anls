# `.\pytorch\test\package\test_package_script.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入所需模块和类
from io import BytesIO
from textwrap import dedent
from unittest import skipIf

# 导入 torch 相关模块和类
import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests

try:
    # 尝试导入本地的 PackageTestCase
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则从全局导入
    # 支持直接运行此文件的情况
    from common import PackageTestCase

try:
    # 尝试导入 torchvision 的 resnet18 模型
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    # 如果导入失败，则设置为 False
    HAS_TORCHVISION = False

# 如果没有导入 torchvision，则设置 skipIfNoTorchVision 为跳过测试
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")

# 定义一个测试类，继承自 PackageTestCase
class TestPackageScript(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    # 定义一个测试方法，测试打包接口类的正确性
    def test_package_interface(self):
        """Packaging an interface class should work correctly."""

        # 导入虚拟接口模块
        import package_a.fake_interface as fake

        # 创建虚拟接口对象
        uses_interface = fake.UsesInterface()
        
        # 对接口进行 TorchScript 编译
        scripted = torch.jit.script(uses_interface)
        
        # 为脚本化的接口设置代理模块
        scripted.proxy_mod = torch.jit.script(fake.NewModule())

        # 创建一个字节流对象
        buffer = BytesIO()

        # 使用 PackageExporter 打包内容
        with PackageExporter(buffer) as pe:
            pe.intern("**")  # 将所有内容打包
            pe.save_pickle("model", "model.pkl", uses_interface)  # 保存打包后的模型

        buffer.seek(0)  # 将文件指针移动到开头，准备读取

        # 创建 PackageImporter 对象，用于导入打包后的内容
        package_importer = PackageImporter(buffer)
        
        # 加载打包后的模型
        loaded = package_importer.load_pickle("model", "model.pkl")

        # 对加载的模型进行 TorchScript 编译
        scripted_loaded = torch.jit.script(loaded)
        
        # 为加载的脚本化模型设置代理模块
        scripted_loaded.proxy_mod = torch.jit.script(fake.NewModule())

        # 创建输入张量
        input = torch.tensor(1)

        # 断言两个脚本化模型在给定输入下的输出是否一致
        self.assertEqual(scripted(input), scripted_loaded(input))
    def test_different_package_interface(self):
        """测试一个情况，其中包中定义的接口与加载环境中定义的接口不同，以确保 TorchScript 能够区分它们。"""
        # 导入一个版本的接口
        import package_a.fake_interface as fake

        # 模拟一个包，其中包含一个具有完全相同名称的不同版本接口。
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch
                    from torch import Tensor

                    @torch.jit.interface
                    class ModuleInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            pass

                    class ImplementsInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            return inp1 + 1

                    class UsesInterface(torch.nn.Module):
                        proxy_mod: ModuleInterface

                        def __init__(self):
                            super().__init__()
                            self.proxy_mod = ImplementsInterface()

                        def forward(self, input: Tensor) -> Tensor:
                            return self.proxy_mod.one(input)
                    """
                ),
            )
        buffer.seek(0)

        # 创建包导入器对象
        package_importer = PackageImporter(buffer)
        # 导入不同版本的 fake 模块
        diff_fake = package_importer.import_module(fake.__name__)
        # 我们应该能够成功地进行脚本化。
        torch.jit.script(diff_fake.UsesInterface())

    def test_package_script_class(self):
        """测试包脚本类的情况。"""
        import package_a.fake_script_class as fake

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(fake.__name__)
        buffer.seek(0)

        # 创建包导入器对象
        package_importer = PackageImporter(buffer)
        # 导入加载的 fake 模块
        loaded = package_importer.import_module(fake.__name__)

        input = torch.tensor(1)
        self.assertTrue(
            torch.allclose(
                fake.uses_script_class(input), loaded.uses_script_class(input)
            )
        )
    def test_package_script_class_referencing_self(self):
        # 导入需要测试的脚本类别
        import package_a.fake_script_class as fake

        # 创建 fake.UsesIdListFeature 的实例对象
        obj = fake.UsesIdListFeature()

        # 利用 TorchScript 将对象进行脚本化，填充编译缓存，确保来自包内和外部环境的脚本化类型之间没有错误共享。
        torch.jit.script(obj)

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将 "**" 转换为内部表示，并保存对象 obj 到 obj.pkl 文件中
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle("obj", "obj.pkl", obj)

        # 将缓冲区指针移动到起始位置
        buffer.seek(0)
        # 创建 PackageImporter 对象来导入数据
        importer = PackageImporter(buffer)
        # 从 obj.pkl 中加载对象 obj_loaded
        obj_loaded = importer.load_pickle("obj", "obj.pkl")
        # 对加载的对象进行 TorchScript 脚本化
        scripted_obj_loaded = torch.jit.script(obj_loaded)

        # 确保脚本化对象可以无错误地进行序列化
        buffer2 = scripted_obj_loaded.save_to_buffer()
        torch.jit.load(BytesIO(buffer2))

    def test_different_package_script_class(self):
        """Test a case where the script class defined in the package is
        different than the one defined in the loading environment, to make
        sure TorchScript can distinguish between the two.
        """
        # 导入需要测试的脚本类别
        import package_a.fake_script_class as fake

        # 模拟包含不同版本脚本类的情况，其中属性为 `bar` 而不是 `foo`
        buffer = BytesIO()
        with PackageExporter(buffer) as pe2:
            pe2.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch

                    @torch.jit.script
                    class MyScriptClass:
                        def __init__(self, x):
                            self.bar = x
                    """
                ),
            )
        buffer.seek(0)

        # 创建 PackageImporter 对象来导入数据
        package_importer = PackageImporter(buffer)
        # 导入不同版本的 fake 脚本类
        diff_fake = package_importer.import_module(fake.__name__)
        input = torch.rand(2, 3)
        # 加载脚本化类 MyScriptClass 的实例
        loaded_script_class = diff_fake.MyScriptClass(input)
        orig_script_class = fake.MyScriptClass(input)
        # 断言加载的类与原始脚本类的属性相同
        self.assertEqual(loaded_script_class.bar, orig_script_class.foo)

    def test_save_scriptmodule(self):
        """
        Test basic saving of ScriptModule.
        """
        # 从 package_a.test_module 导入 ModWithTensor 类
        from package_a.test_module import ModWithTensor

        # 对 ModWithTensor 类的实例进行 TorchScript 脚本化
        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将脚本模块保存为 mod.pkl 文件
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        # 将缓冲区指针移动到起始位置
        buffer.seek(0)
        # 创建 PackageImporter 对象来导入数据
        importer = PackageImporter(buffer)
        # 从 mod.pkl 中加载模块
        loaded_mod = importer.load_pickle("res", "mod.pkl", map_location="cpu")
        input = torch.rand(1, 2, 3)
        # 断言加载的模块与脚本化模块的行为相同
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_save_scriptmodule_file(self):
        """
        Test basic saving of ScriptModule in file.
        """
        # 从 package_a.test_module 模块导入 ModWithTensor 类
        from package_a.test_module import ModWithTensor
        
        # 使用 torch.jit.script 方法对 ModWithTensor 类的实例进行脚本化
        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        
        # 创建临时文件名
        filename = self.temp()
        
        # 使用 PackageExporter 类创建文件导出对象 e，并打开文件 filename
        with PackageExporter(filename) as e:
            # 将 scripted_mod 对象以 Pickle 格式保存为 "res/mod.pkl"
            e.save_pickle("res", "mod.pkl", scripted_mod)
        
        # 使用 PackageImporter 类创建文件导入对象 importer，加载 "res/mod.pkl" 文件
        importer = PackageImporter(filename)
        # 从 importer 中加载 "res/mod.pkl" 文件，得到 loaded_mod 对象
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        
        # 创建输入数据 input
        input = torch.rand(1, 2, 3)
        
        # 断言 loaded_mod 对象对 input 的计算结果与 scripted_mod 对象对 input 的计算结果相等
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    def test_save_scriptmodule_with_submods(self):
        """
        Test basic saving of ScriptModule with submodule.
        """
        # 从 package_a.test_module 模块导入 ModWithSubmod 和 ModWithTensor 类
        from package_a.test_module import ModWithSubmod, ModWithTensor
        
        # 使用 torch.jit.script 方法对 ModWithTensor 类的实例进行脚本化，并作为 ModWithSubmod 类的参数
        scripted_mod = torch.jit.script(
            ModWithSubmod(ModWithTensor(torch.rand(1, 2, 3)))
        )
        
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        
        # 使用 PackageExporter 类创建字节流导出对象 e
        with PackageExporter(buffer) as e:
            # 将 scripted_mod 对象以 Pickle 格式保存为 "res/mod.pkl"，存储在 buffer 中
            e.save_pickle("res", "mod.pkl", scripted_mod)
        
        # 将 buffer 的指针移动到开始位置
        buffer.seek(0)
        
        # 使用 PackageImporter 类创建从 buffer 加载的导入对象 importer
        importer = PackageImporter(buffer)
        # 从 importer 中加载 "res/mod.pkl" 文件，使用 "cpu" 作为 map_location 参数
        loaded_mod = importer.load_pickle("res", "mod.pkl", map_location="cpu")
        
        # 创建输入数据 input
        input = torch.rand(1, 2, 3)
        
        # 断言 loaded_mod 对象对 input 的计算结果与 scripted_mod 对象对 input 的计算结果相等
        self.assertEqual(loaded_mod(input), scripted_mod(input))
    def test_save_scriptmodules_submod_redefinition(self):
        """
        Test to verify saving multiple ScriptModules with same top module
        but different submodules works. Submodule is redefined to between
        the defintion of the top module to check that the different concrete
        types of the modules are thoroughly recognized by serializaiton code.
        """

        # 定义一个名为 Submod 的内部类，继承自 torch.nn.Module
        class Submod(torch.nn.Module):
            # 重写 forward 方法，对输入的字符串进行处理
            def forward(self, input: str):
                input = input + "_submod"
                return input

        # 定义一个名为 TopMod 的内部类，继承自 torch.nn.Module
        class TopMod(torch.nn.Module):
            # 构造方法，初始化时创建 Submod 实例并赋给 modB
            def __init__(self):
                super().__init__()
                self.modB = Submod()

            # 重写 forward 方法，调用 modB 的 forward 方法处理输入字符串
            def forward(self, input: str):
                return self.modB(input)

        # 使用 torch.jit.script 将 TopMod 实例 script 化，得到 scripted_mod_0
        scripted_mod_0 = torch.jit.script(TopMod())

        # 故意重新定义 Submod 类，改变其 forward 方法的处理逻辑，应触发新的模块类型
        class Submod(torch.nn.Module):  # noqa: F811
            def forward(self, input: str):
                input = input + "_submod(changed)"
                return input

        # 再次使用 torch.jit.script 将 TopMod 实例 script 化，得到 scripted_mod_1
        scripted_mod_1 = torch.jit.script(TopMod())

        # 创建一个 BytesIO 对象 buffer，用于存储序列化后的模型数据
        buffer = BytesIO()

        # 使用 PackageExporter 将两个 script 化的模型保存到 buffer 中
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        # 将 buffer 的指针位置移到开头
        buffer.seek(0)

        # 使用 PackageImporter 从 buffer 中导入模型数据
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")

        # 断言加载的两个模型在相同输入下与原始的 script 化模型行为一致
        self.assertEqual(loaded_mod_0("input"), scripted_mod_0("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_1("input"))

        # 断言加载的两个模型在相同输入下行为不同，验证模型已被正确序列化和反序列化
        self.assertNotEqual(loaded_mod_0("input"), loaded_mod_1("input"))

    def test_save_independent_scriptmodules(self):
        """
        Test to verify saving multiple ScriptModules with completely
        separate code works.
        """
        # 从 package_a.test_module 中导入 ModWithTensor 和 SimpleTest 类
        from package_a.test_module import ModWithTensor, SimpleTest

        # 使用 torch.jit.script 将 SimpleTest 实例 script 化，得到 scripted_mod_0
        scripted_mod_0 = torch.jit.script(SimpleTest())

        # 创建一个包含随机张量的 ModWithTensor 实例，并使用 torch.jit.script 将其 script 化，得到 scripted_mod_1
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        # 创建一个 BytesIO 对象 buffer，用于存储序列化后的模型数据
        buffer = BytesIO()

        # 使用 PackageExporter 将两个 script 化的模型保存到 buffer 中
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        # 将 buffer 的指针位置移到开头
        buffer.seek(0)

        # 使用 PackageImporter 从 buffer 中导入模型数据
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")

        # 创建一个随机张量作为输入
        input = torch.rand(1, 2, 3)

        # 断言加载的两个模型在相同输入下与原始的 script 化模型行为一致
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))
    def test_save_repeat_scriptmodules(self):
        """
        Test to verify saving multiple different modules and
        repeats of same scriptmodule in package works. Also tests that
        PyTorchStreamReader isn't having code hidden from
        PyTorchStreamWriter writing ScriptModule code files multiple times.
        """
        # 导入需要测试的模块
        from package_a.test_module import (
            ModWithSubmodAndTensor,
            ModWithTensor,
            SimpleTest,
        )

        # 对 SimpleTest 类进行 Torch 脚本化
        scripted_mod_0 = torch.jit.script(SimpleTest())
        # 创建包含随机张量的 ModWithTensor 类的 Torch 脚本
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        # 创建包含子模块和随机张量的 ModWithSubmodAndTensor 类的 Torch 脚本
        scripted_mod_2 = torch.jit.script(
            ModWithSubmodAndTensor(
                torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))
            )
        )

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将脚本化的模块保存到字节流缓冲区中
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod0.pkl", scripted_mod_0)
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)
            e.save_pickle("res", "mod2.pkl", scripted_mod_0)  # 重复保存 mod0
            e.save_pickle("res", "mod3.pkl", scripted_mod_1)  # 重复保存 mod1
            e.save_pickle("res", "mod4.pkl", scripted_mod_2)

        # 将字节流缓冲区的读取位置设置为起始位置
        buffer.seek(0)
        # 创建 PackageImporter 实例来从字节流缓冲区中加载保存的模块
        importer = PackageImporter(buffer)
        # 加载保存的模块
        loaded_mod_0 = importer.load_pickle("res", "mod0.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod3.pkl")
        loaded_mod_2 = importer.load_pickle("res", "mod4.pkl")
        # 创建输入张量
        input = torch.rand(1, 2, 3)
        # 断言加载的模块和脚本化的模块在相同输入下的输出是否相等
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))
        self.assertEqual(loaded_mod_2(input), scripted_mod_2(input))
    def test_scriptmodules_repeat_save(self):
        """
        Test to verify saving and loading same ScriptModule object works
        across multiple packages.
        """
        # 导入必要的模块和类
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor
        
        # 创建第一个脚本化的模块对象
        scripted_mod_0 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        
        # 创建第二个脚本化的模块对象，其中包含子模块和张量
        scripted_mod_1 = torch.jit.script(
            ModWithSubmodAndTensor(
                torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))
            )
        )

        # 创建第一个缓冲区对象
        buffer_0 = BytesIO()
        # 使用 PackageExporter 将第一个脚本化模块保存到缓冲区中
        with PackageExporter(buffer_0) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)

        # 将缓冲区指针移动到起始位置
        buffer_0.seek(0)
        # 创建 PackageImporter 对象来导入第一个缓冲区中的内容
        importer_0 = PackageImporter(buffer_0)
        # 从导入的内容中加载 mod1.pkl 文件
        loaded_module_0 = importer_0.load_pickle("res", "mod1.pkl")

        # 创建第二个缓冲区对象
        buffer_1 = BytesIO()
        # 使用 PackageExporter 将第二个脚本化模块和第一个加载的模块保存到缓冲区中
        with PackageExporter(buffer_1) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)
            e.save_pickle("res", "mod2.pkl", loaded_module_0)

        # 将缓冲区指针移动到起始位置
        buffer_1.seek(0)
        # 创建 PackageImporter 对象来导入第二个缓冲区中的内容
        importer_1 = PackageImporter(buffer_1)
        # 从导入的内容中加载 mod1.pkl 文件
        loaded_module_1 = importer_1.load_pickle("res", "mod1.pkl")
        # 从导入的内容中加载 mod2.pkl 文件
        reloaded_module_0 = importer_1.load_pickle("res", "mod2.pkl")

        # 创建输入张量
        input = torch.rand(1, 2, 3)
        # 断言加载的模块与原始脚本化模块在相同输入下的输出一致
        self.assertEqual(loaded_module_0(input), scripted_mod_0(input))
        # 断言重新加载的模块与第一个加载的模块在相同输入下的输出一致
        self.assertEqual(loaded_module_0(input), reloaded_module_0(input))
        # 断言加载的第二个模块与原始脚本化模块在相同输入下的输出一致
        self.assertEqual(loaded_module_1(input), scripted_mod_1(input))

    @skipIfNoTorchVision
    def test_save_scriptmodule_only_necessary_code(self):
        """
        Test to verify when saving multiple packages with same CU
        that packages don't include unnecessary torchscript code files.
        The TorchVision code should only be saved in the package that
        relies on it.
        """
        # 导入必要的模块和类
        from package_a.test_module import ModWithTensor
        
        # 定义一个包含 TorchVision 的模块类
        class ModWithTorchVision(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.tvmod = resnet18()

            def forward(self, input):
                return input * 4

        # 创建第一个脚本化的模块对象，其中包含 TorchVision
        scripted_mod_0 = torch.jit.script(ModWithTorchVision("foo"))
        # 创建第二个脚本化的模块对象，仅包含普通张量模块
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        # 创建第一个缓冲区对象
        buffer_0 = BytesIO()
        # 使用 PackageExporter 将第一个脚本化模块保存到缓冲区中
        with PackageExporter(buffer_0) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)

        # 将缓冲区指针移动到起始位置
        buffer_0.seek(0)
        # 创建 PackageImporter 对象来导入第一个缓冲区中的内容
        importer_0 = PackageImporter(buffer_0)

        # 创建第二个缓冲区对象
        buffer_1 = BytesIO()
        # 使用 PackageExporter 将第二个脚本化模块保存到缓冲区中
        with PackageExporter(buffer_1) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)

        # 将缓冲区指针移动到起始位置
        buffer_1.seek(0)
        # 创建 PackageImporter 对象来导入第二个缓冲区中的内容
        importer_1 = PackageImporter(buffer_1)

        # 断言第一个缓冲区中包含 TorchVision 相关文件
        self.assertTrue("torchvision" in str(importer_0.file_structure()))
        # 断言第二个缓冲区中不包含 TorchVision 相关文件
        self.assertFalse("torchvision" in str(importer_1.file_structure()))
    def test_save_scriptmodules_in_container(self):
        """
        Test saving of ScriptModules inside of container. Checks that relations
        between shared modules are upheld.
        """
        # 导入需要测试的模块
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor

        # 创建一个脚本化的 ScriptModule，包含一个张量作为参数
        scripted_mod_a = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        # 创建另一个脚本化的 ScriptModule，包含一个子模块和一个张量作为参数
        scripted_mod_b = torch.jit.script(
            ModWithSubmodAndTensor(torch.rand(1, 2, 3), scripted_mod_a)
        )
        # 将这两个 ScriptModule 放入列表中
        script_mods_list = [scripted_mod_a, scripted_mod_b]

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将脚本化的模块列表保存为 pickle 文件
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "list.pkl", script_mods_list)

        # 将缓冲区指针设置回开头
        buffer.seek(0)
        # 创建一个 PackageImporter 实例来导入保存的 pickle 文件
        importer = PackageImporter(buffer)
        # 从导入的文件中加载模块列表
        loaded_mod_list = importer.load_pickle("res", "list.pkl")
        # 创建一个输入张量
        input = torch.rand(1, 2, 3)
        # 断言加载的模块能够正确处理输入
        self.assertEqual(loaded_mod_list[0](input), scripted_mod_a(input))
        self.assertEqual(loaded_mod_list[1](input), scripted_mod_b(input))

    def test_save_eager_mods_sharing_scriptmodule(self):
        """
        Test saving of single ScriptModule shared by multiple
        eager modules (ScriptModule should be saved just once
        even though is contained in multiple pickles).
        """
        # 导入需要测试的模块
        from package_a.test_module import ModWithSubmod, SimpleTest

        # 创建一个脚本化的 ScriptModule
        scripted_mod = torch.jit.script(SimpleTest())

        # 创建两个包含相同 ScriptModule 的模块
        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将模块保存为 pickle 文件，同时内部化所有内容
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", mod1)
            e.save_pickle("res", "mod2.pkl", mod2)

        # 将缓冲区指针设置回开头
        buffer.seek(0)
        # 创建一个 PackageImporter 实例来导入保存的 pickle 文件
        importer = PackageImporter(buffer)
        # 检查导入的文件结构
        file_structure = importer.file_structure()
        # 断言是否正确保存了共享的 ScriptModule
        self.assertTrue(file_structure.has_file(".data/ts_code/0"))
        self.assertFalse(file_structure.has_file(".data/ts_code/1"))

    def test_load_shared_scriptmodules(self):
        """
        Test loading of single ScriptModule shared by multiple eager
        modules in single pickle (ScriptModule objects should be the same).
        """
        # 导入需要测试的模块
        from package_a.test_module import (
            ModWithMultipleSubmods,
            ModWithSubmod,
            SimpleTest,
        )

        # 创建一个脚本化的 ScriptModule
        scripted_mod = torch.jit.script(SimpleTest())

        # 创建两个包含相同 ScriptModule 的模块
        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)

        # 创建一个父模块，包含这两个子模块
        mod_parent = ModWithMultipleSubmods(mod1, mod2)

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将模块保存为 pickle 文件，同时内部化所有内容
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "mod.pkl", mod_parent)

        # 将缓冲区指针设置回开头
        buffer.seek(0)
        # 创建一个 PackageImporter 实例来导入保存的 pickle 文件
        importer = PackageImporter(buffer)

        # 从导入的文件中加载模块
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        # 断言加载的模块中共享的 ScriptModule 对象是相同的
        self.assertTrue(
            id(loaded_mod.mod1.script_mod) == id(loaded_mod.mod2.script_mod)
        )
    def test_save_shared_tensors(self):
        """
        Test tensors shared across eager and ScriptModules are serialized once.
        """
        # 导入需要测试的模块和类
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor

        # 创建一个共享的随机张量
        shared_tensor = torch.rand(2, 3, 4)

        # 对带有张量的模块进行脚本化
        scripted_mod = torch.jit.script(ModWithTensor(shared_tensor))

        # 创建两个实例，共享相同的张量和脚本化模块
        mod1 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)
        mod2 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)

        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 使用PackageExporter将对象保存到缓冲区
        with PackageExporter(buffer) as e:
            # 将所有内容都保存在资源名为"res"下
            e.intern("**")
            # 将共享的张量保存为pickle对象
            e.save_pickle("res", "tensor", shared_tensor)
            # 将mod1和mod2分别保存为pickle对象
            e.save_pickle("res", "mod1.pkl", mod1)
            e.save_pickle("res", "mod2.pkl", mod2)

        # 将缓冲区的指针移到开头
        buffer.seek(0)

        # 创建一个PackageImporter来导入保存的数据
        importer = PackageImporter(buffer)

        # 加载mod1.pkl文件
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        # 断言：确保包中只存储了一个存储对象
        file_structure = importer.file_structure(include=".data/*.storage")
        self.assertTrue(len(file_structure.children[".data"].children) == 1)

        # 创建一个随机输入张量
        input = torch.rand(2, 3, 4)

        # 断言：加载后的模块能够正确处理输入
        self.assertEqual(loaded_mod_1(input), mod1(input))

    def test_load_shared_tensors(self):
        """
        Test tensors shared across eager and ScriptModules on load
        are the same.
        """
        # 导入需要测试的模块和类
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor

        # 创建一个共享的全1张量
        shared_tensor = torch.ones(3, 3)

        # 对两个带张量的模块进行脚本化
        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))

        # 创建一个模块实例，包含两个子模块和一个共享的张量
        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)

        # 断言：确保两个脚本化模块共享相同的张量存储
        self.assertEqual(
            shared_tensor.storage()._cdata,
            scripted_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            shared_tensor.storage()._cdata,
            scripted_mod_1.tensor.storage()._cdata,
        )

        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 使用PackageExporter将模块保存到缓冲区
        with PackageExporter(buffer) as e:
            # 将所有内容都保存在资源名为"res"下
            e.intern("**")
            # 将mod1保存为pickle对象
            e.save_pickle("res", "mod1.pkl", mod1)

        # 将缓冲区的指针移到开头
        buffer.seek(0)

        # 创建一个PackageImporter来导入保存的数据
        importer = PackageImporter(buffer)

        # 加载mod1.pkl文件
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        # 断言：加载后的模块仍然共享相同的张量存储
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_1.tensor.storage()._cdata,
        )

        # 对加载后的模块进行操作，并检查是否影响了子模块的张量
        loaded_mod_1.tensor.add_(torch.ones(3, 3))

        # 断言：加载后的模块的子模块张量应与加载后的模块的张量相等
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor)
        )
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor)
        )
    def test_load_shared_tensors_repackaged(self):
        """
        Test tensors shared across eager and ScriptModules on load
        are the same across multiple package saves and loads. This is
        an important test because not all of the tensor information is restored
        in python between packages. The python identity is not maintained, but
        the backing cpp TensorImpl is. We load/save storages based off of this
        cpp TensorImpl and not the python identity.
        """
        # 导入需要测试的模块
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor

        # 创建一个3x3的全1张量
        shared_tensor = torch.ones(3, 3)

        # 对模块进行脚本化
        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))

        # 创建包含共享张量的复合模块
        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)

        # 创建一个字节流缓冲区
        buffer_0 = BytesIO()
        # 使用PackageExporter将模块保存到缓冲区
        with PackageExporter(buffer_0) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", mod1)

        # 将缓冲区指针重置到开头
        buffer_0.seek(0)
        # 使用PackageImporter加载保存的模块
        importer_0 = PackageImporter(buffer_0)
        loaded_mod_0 = importer_0.load_pickle("res", "mod1.pkl")

        # 创建第二个字节流缓冲区
        buffer_1 = BytesIO()
        # 使用PackageExporter将已加载的模块再次保存到新缓冲区中
        with PackageExporter(buffer_1, importer=importer_0) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", loaded_mod_0)

        # 将缓冲区指针重置到开头
        buffer_1.seek(0)
        # 使用PackageImporter加载第二次保存的模块
        importer = PackageImporter(buffer_1)
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        # 断言两个子模块共享的张量在存储层面相同
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_1.tensor.storage()._cdata,
        )

        # 修改主模块的张量，期望所有相关的张量都反映这一变化
        loaded_mod_1.tensor.add_(torch.ones(3, 3))

        # 断言所有相关张量是否相等
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor)
        )
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor)
        )
    def test_mixing_packaged_and_inline_modules(self):
        """
        Test saving inline and imported modules in same package with
        independent code.
        """

        # 定义一个内联模块，继承自 torch.nn.Module
        class InlineMod(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 2, 3)

            # 前向传播函数，接受一个字符串输入并返回处理后的字符串和张量乘以4的结果
            def forward(self, input: str):
                input = input + "_modInline:" + self.name
                return input, (self.tensor * 4)

        # 创建一个名为 'inline' 的内联模块实例
        inline_mod = InlineMod("inline")

        # 对内联模块进行脚本化
        scripted_inline = torch.jit.script(inline_mod)

        # 从 package_a.test_module 模块中导入 SimpleTest 类
        from package_a.test_module import SimpleTest

        # 创建一个 SimpleTest 类的实例 imported_mod
        imported_mod = SimpleTest()

        # 对 imported_mod 进行脚本化
        scripted_imported = torch.jit.script(imported_mod)

        # 创建一个字节流缓冲区 buffer
        buffer = BytesIO()

        # 使用 PackageExporter 对象 e，将脚本化的 inline_mod 和 imported_mod 保存到 buffer 中
        with PackageExporter(buffer) as e:
            e.save_pickle("model", "inline.pkl", scripted_inline)  # 将脚本化的 inline_mod 保存为 'model/inline.pkl'
            e.save_pickle("model", "imported.pkl", scripted_imported)  # 将脚本化的 imported_mod 保存为 'model/imported.pkl'

        # 将缓冲区指针移动到起始位置
        buffer.seek(0)

        # 创建一个 PackageImporter 对象 importer，用于从 buffer 中导入数据
        importer = PackageImporter(buffer)

        # 从 importer 中加载 'model/inline.pkl' 并赋值给 loaded_inline
        loaded_inline = importer.load_pickle("model", "inline.pkl")

        # 从 importer 中加载 'model/imported.pkl' 并赋值给 loaded_imported
        loaded_imported = importer.load_pickle("model", "imported.pkl")

        # 创建一个大小为 (2, 3) 的随机张量输入
        input = torch.rand(2, 3)

        # 断言 loaded_imported(input) 结果与 imported_mod(input) 相等
        self.assertEqual(loaded_imported(input), imported_mod(input))

        # 断言 loaded_inline("input") 结果与 inline_mod("input") 相等
        self.assertEqual(loaded_inline("input"), inline_mod("input"))
    def test_mixing_packaged_and_inline_modules_shared_code(self):
        """
        Test saving inline and imported modules in same package that
        share code.
        """

        class TorchVisionTestInline(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的内联模块
            def __init__(self):
                super().__init__()
                # 在构造函数中初始化一个 torchvision 的 ResNet-18 模型
                self.tvmod = resnet18()

            # 定义前向传播函数
            def forward(self, x):
                # 调用一个非 PyTorch 的函数 a_non_torch_leaf 处理输入
                x = a_non_torch_leaf(x, x)
                # 对处理后的结果执行 ReLU 激活函数
                return torch.relu(x + 3.0)

        # 定义一个非 PyTorch 函数 a_non_torch_leaf，用于在模块中调用
        def a_non_torch_leaf(a, b):
            return a + b

        # 创建一个 TorchVisionTestInline 的实例
        inline_mod = TorchVisionTestInline()
        # 将内联模块进行脚本化
        scripted_inline = torch.jit.script(inline_mod)

        # 从 package_c.test_module 导入 TorchVisionTest 类
        from package_c.test_module import TorchVisionTest

        # 创建一个 TorchVisionTest 的实例
        imported_mod = TorchVisionTest()
        # 将导入的模块进行脚本化
        scripted_imported = torch.jit.script(imported_mod)

        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将脚本化的模块保存到 buffer 中
        with PackageExporter(buffer) as e:
            e.save_pickle("model", "inline.pkl", scripted_inline)
            e.save_pickle("model", "imported.pkl", scripted_imported)

        # 将 buffer 的读写位置移动到开头
        buffer.seek(0)
        # 使用 PackageImporter 从 buffer 中导入模块
        importer = PackageImporter(buffer)
        # 加载从 buffer 中保存的 inline.pkl
        loaded_inline = importer.load_pickle("model", "inline.pkl")
        # 加载从 buffer 中保存的 imported.pkl
        loaded_imported = importer.load_pickle("model", "imported.pkl")

        # 创建一个随机输入张量
        input = torch.rand(2, 3)
        # 断言加载的 imported.pkl 的结果与原始 imported_mod(input) 的结果相同
        self.assertEqual(loaded_imported(input), imported_mod(input))
        # 断言加载的 inline.pkl 的结果与原始 inline_mod(input) 的结果相同
        self.assertEqual(loaded_inline(input), inline_mod(input))

    def test_tensor_sharing_pickle(self):
        """Test that saving a ScriptModule and separately saving a tensor
        object causes no issues.
        """

        # 定义一个简单的 PyTorch 模块 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个大小为 (2, 3) 的全为 1 的张量
                self.foo = torch.ones(2, 3)

            # 定义模块的前向传播函数
            def forward(self):
                return self.foo

        # 对模块 M 进行脚本化
        scripted_m = torch.jit.script(M())
        # 创建一个空张量
        original_tensor = torch.ones(0)

        # 创建一个字节流对象 f
        f = BytesIO()
        # 使用 PackageExporter 将脚本化的模块和张量分别保存到 f 中
        with torch.package.PackageExporter(f) as exporter:
            exporter.save_pickle("model", "model.pkl", scripted_m)
            exporter.save_pickle("model", "input.pkl", original_tensor)

        # 将 f 的读写位置移动到开头
        f.seek(0)
        # 使用 PackageImporter 从 f 中导入模块和张量
        importer = PackageImporter(f)
        # 加载从 f 中保存的 model.pkl
        loaded_m = importer.load_pickle("model", "model.pkl")
        # 加载从 f 中保存的 input.pkl
        loaded_tensor = importer.load_pickle("model", "input.pkl")

        # 断言加载的模块的 foo 属性与原始模块的 foo 属性相同
        self.assertEqual(scripted_m.foo, loaded_m.foo)
        # 断言加载的张量与原始张量相同
        self.assertEqual(original_tensor, loaded_tensor)
# 如果当前脚本被直接执行（而不是被导入到其它模块中），则执行以下代码块
if __name__ == "__main__":
    # 调用一个名为 run_tests() 的函数，用于执行测试或其它相关的任务
    run_tests()
```