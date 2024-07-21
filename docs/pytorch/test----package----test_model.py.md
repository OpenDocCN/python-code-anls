# `.\pytorch\test\package\test_model.py`

```
# 从 io 模块导入 BytesIO 类，用于操作二进制数据的内存缓冲区
# 从 textwrap 模块导入 dedent 函数，用于去除多行字符串的缩进
# 从 unittest 模块导入 skipIf 装饰器，根据条件跳过测试用例

# 导入 torch 库
import torch
# 从 torch.package 模块导入 PackageExporter, PackageImporter, sys_importer
from torch.package import PackageExporter, PackageImporter, sys_importer
# 从 torch.testing._internal.common_utils 模块导入 IS_FBCODE, IS_SANDCASTLE, run_tests 函数
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests

# 尝试导入 torchvision.models 模块中的 resnet18 模型
try:
    from torchvision.models import resnet18
    # 设置标志指示成功导入 torchvision
    HAS_TORCHVISION = True
except ImportError:
    # 设置标志指示未能导入 torchvision
    HAS_TORCHVISION = False
# 定义 skipIfNoTorchVision 装饰器，如果没有导入 torchvision 则跳过测试
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")

# 尝试导入 .common 模块中的 PackageTestCase 类
try:
    from .common import PackageTestCase
except ImportError:
    # 支持直接运行该文件的情况，导入 common 模块中的 PackageTestCase 类
    from common import PackageTestCase

# 定义 ModelTest 类，继承自 PackageTestCase，用于封装整个模型的端到端测试
@skipIf(
    True,
    "Does not work with recent torchvision, see https://github.com/pytorch/pytorch/issues/81115",
)
@skipIfNoTorchVision
class ModelTest(PackageTestCase):
    """End-to-end tests packaging an entire model."""

    # 装饰器，根据环境条件跳过测试，如在 FBCODE 或 SANDCASTLE 环境下禁用使用临时文件的测试
    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    # 定义一个测试用例，用于测试 ResNet 模型的功能
    def test_resnet(self):
        # 创建一个 ResNet18 模型实例
        resnet = resnet18()

        # 调用 self.temp() 方法，返回临时文件对象 f1
        f1 = self.temp()

        # 使用 PackageExporter 类，将模型 resnet 和其相关的代码保存到 f1 中
        with PackageExporter(f1) as e:
            # 将经过 pickle 处理的 resnet 模型放入包中，默认情况下还会保存所有被 pickle 对象引用的代码文件
            e.intern("**")
            e.save_pickle("model", "model.pkl", resnet)

        # 可以加载保存的模型
        i = PackageImporter(f1)
        r2 = i.load_pickle("model", "model.pkl")

        # 测试模型是否正常工作
        input = torch.rand(1, 3, 224, 224)
        ref = resnet(input)
        self.assertEqual(r2(input), ref)

        # 还可以使用函数访问每个包中的私有模块
        torchvision = i.import_module("torchvision")

        # 创建一个字节流对象 f2
        f2 = BytesIO()

        # 如果进行迁移学习，可能需要重新保存从包中加载的内容
        # 需要告诉导出器关于来自导入包的任何模块，以便解析类名如 torchvision.models.resnet.ResNet 到其源代码
        with PackageExporter(f2, importer=(i, sys_importer)) as e:
            # e.importers 是一个模块导入函数的列表，默认包含 importlib.import_module
            # 它按顺序搜索，直到找到第一个成功的模块，并认为它是当前代码包中 torchvision.models.resnet 的源码位置。
            # 在名称冲突的情况下（如尝试保存来自两个不同包的 ResNet 对象），只有路径中找到的第一个 ResNet 对象可用。
            # 这避免了源代码中的大量名称混淆。如果需要混合使用 ResNet 对象，建议使用像 save_state_dict 和 load_state_dict 这样的函数，
            # 使用单个包中的代码对象重建模型对象。
            e.intern("**")
            e.save_pickle("model", "model.pkl", r2)

        # 将 f2 的读取位置移至开头
        f2.seek(0)

        # 创建 PackageImporter 实例 i2，用于导入 f2 中的内容
        i2 = PackageImporter(f2)
        r3 = i2.load_pickle("model", "model.pkl")

        # 测试从 f2 中加载的模型是否与原模型功能相同
        self.assertEqual(r3(input), ref)

    # 使用装饰器标记，如果没有安装 torchvision 则跳过此测试
    @skipIfNoTorchVision
    @skipIfNoTorchVision
    # 定义一个测试函数，测试对 ResNet 模型的脚本化和序列化操作
    def test_script_resnet(self):
        # 创建一个 ResNet-18 模型实例
        resnet = resnet18()

        # 创建一个字节流对象 f1
        f1 = BytesIO()

        # 选项1：使用 pickle 将整个模型保存下来
        # + 单行代码，类似于 torch.jit.save
        # - 在创建模型后编辑代码更为困难
        # 使用 PackageExporter 对象 e，将模型内部的所有内容导出到 f1 中
        with PackageExporter(f1) as e:
            e.intern("**")
            e.save_pickle("model", "pickled", resnet)

        # 将 f1 的读写位置移动到开头
        f1.seek(0)

        # 创建 PackageImporter 对象 i，用于从 f1 中导入数据
        i = PackageImporter(f1)

        # 从 f1 中加载名为 "model"、类型为 "pickled" 的数据
        loaded = i.load_pickle("model", "pickled")

        # 将加载的模型进行脚本化
        scripted = torch.jit.script(loaded)

        # 将脚本化后的模型保存到字节流对象 f2 中
        f2 = BytesIO()
        torch.jit.save(scripted, f2)

        # 将 f2 的读写位置移动到开头
        f2.seek(0)

        # 从 f2 中加载模型
        loaded = torch.jit.load(f2)

        # 创建一个随机输入
        input = torch.rand(1, 3, 224, 224)

        # 断言加载后的模型对输入的输出与原始模型 resnet 对输入的输出一致
        self.assertEqual(loaded(input), resnet(input))
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则运行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```