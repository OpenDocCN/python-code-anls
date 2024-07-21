# `.\pytorch\test\jit\test_scriptmod_ann.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块和库
import os
import sys
import unittest
import warnings
from typing import Dict, List, Optional

import torch

# 将测试文件夹中的辅助文件设置为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行该脚本，抛出运行时错误，提示正确的使用方法
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自JitTestCase，用于测试脚本模块实例属性类型注解
class TestScriptModuleInstanceAttributeTypeAnnotation(JitTestCase):

    # NB: 这里没有测试 `Tuple` 或 `NamedTuple`。实际上，
    # 将非空的元组重新分配给先前类型为空元组的属性应该失败。参见 `_check.py` 中的说明

    # 测试注解为空的基本类型
    def test_annotated_falsy_base_type(self):
        # 定义一个简单的Module子类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个整数类型的属性 x
                self.x: int = 0

            def forward(self, x: int):
                # 在 forward 方法中将属性 x 赋值为输入的 x
                self.x = x
                return 1

        # 捕获警告并验证警告列表为空
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (1,))
        assert len(w) == 0

    # 测试注解为非空容器类型的情况
    def test_annotated_nonempty_container(self):
        # 定义一个Module子类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个列表类型的属性 x
                self.x: List[int] = [1, 2, 3]

            def forward(self, x: List[int]):
                # 在 forward 方法中将属性 x 赋值为输入的 x
                self.x = x
                return 1

        # 捕获警告并验证警告列表为空
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    # 测试注解为空张量类型的情况
    def test_annotated_empty_tensor(self):
        # 定义一个Module子类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个空张量类型的属性 x
                self.x: torch.Tensor = torch.empty(0)

            def forward(self, x: torch.Tensor):
                # 在 forward 方法中将属性 x 赋值为输入的 x
                self.x = x
                return self.x

        # 捕获警告并验证警告列表为空
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (torch.rand(2, 3),))
        assert len(w) == 0

    # 测试带有jit属性的注解情况
    def test_annotated_with_jit_attribute(self):
        # 定义一个Module子类
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个具有jit属性的空列表类型的属性 x
                self.x = torch.jit.Attribute([], List[int])

            def forward(self, x: List[int]):
                # 在 forward 方法中将属性 x 赋值为输入的 x
                self.x = x
                return self.x

        # 捕获警告并验证警告列表为空
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    # 测试仅在类级别进行注解的情况
    def test_annotated_class_level_annotation_only(self):
        # 定义一个Module子类
        class M(torch.nn.Module):
            # 类级别的注解，类型为列表类型的整数
            x: List[int]

            def __init__(self):
                super().__init__()
                # 初始化一个空列表类型的属性 x
                self.x = []

            def forward(self, y: List[int]):
                # 在 forward 方法中将属性 x 赋值为输入的 y
                self.x = y
                return self.x

        # 捕获警告并验证警告列表为空
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0
    def test_annotated_class_level_annotation_and_init_annotation(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 类级别的类型注解，x 是一个整数列表
            x: List[int]

            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 实例级别的类型注解，self.x 初始化为空列表
                self.x: List[int] = []

            # 前向传播方法，接受一个整数列表 y 作为输入
            def forward(self, y: List[int]):
                # 将输入的 y 赋值给实例变量 self.x
                self.x = y
                # 返回赋值后的 self.x
                return self.x

        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 调用 self.checkModule 方法，传入 M 的实例和一个元组 ([1, 2, 3],)
            self.checkModule(M(), ([1, 2, 3],))
        # 断言没有捕获到任何警告信息
        assert len(w) == 0

    def test_annotated_class_level_jit_annotation(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 类级别的类型注解，x 是一个整数列表
            x: List[int]

            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 使用 torch.jit.annotate 进行类型注解，self.x 初始化为空的整数列表
                self.x: List[int] = torch.jit.annotate(List[int], [])

            # 前向传播方法，接受一个整数列表 y 作为输入
            def forward(self, y: List[int]):
                # 将输入的 y 赋值给实例变量 self.x
                self.x = y
                # 返回赋值后的 self.x
                return self.x

        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 调用 self.checkModule 方法，传入 M 的实例和一个元组 ([1, 2, 3],)
            self.checkModule(M(), ([1, 2, 3],))
        # 断言没有捕获到任何警告信息
        assert len(w) == 0

    def test_annotated_empty_list(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 实例级别的类型注解，self.x 初始化为空列表
                self.x: List[int] = []

            # 前向传播方法，接受一个整数列表 x 作为输入
            def forward(self, x: List[int]):
                # 将输入的 x 赋值给实例变量 self.x
                self.x = x
                # 返回固定值 1
                return 1

        # 使用 self.assertRaisesRegexWithHighlight 断言捕获的异常信息中包含特定字符串，并带有高亮显示的关键字
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 使用 self.assertWarnsRegex 断言捕获的警告信息中包含特定字符串
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 使用 torch.jit.script 尝试对 M 进行脚本化
                torch.jit.script(M())

    @unittest.skipIf(
        sys.version_info[:2] < (3, 9), "Requires lowercase static typing (Python 3.9+)"
    )
    def test_annotated_empty_list_lowercase(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 实例级别的类型注解，self.x 初始化为空的整数列表
                self.x: list[int] = []

            # 前向传播方法，接受一个整数列表 x 作为输入
            def forward(self, x: list[int]):
                # 将输入的 x 赋值给实例变量 self.x
                self.x = x
                # 返回固定值 1
                return 1

        # 使用 self.assertRaisesRegexWithHighlight 断言捕获的异常信息中包含特定字符串，并带有高亮显示的关键字
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 使用 self.assertWarnsRegex 断言捕获的警告信息中包含特定字符串
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 使用 torch.jit.script 尝试对 M 进行脚本化
                torch.jit.script(M())
    def test_annotated_empty_dict(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置实例属性 self.x 为一个空的字典，键为字符串，值为整数
            def __init__(self):
                super().__init__()
                self.x: Dict[str, int] = {}

            # 前向传播函数，接受参数 x，将其赋值给 self.x，并返回整数 1
            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        # 使用 assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常，并显示自定义消息和高亮的代码行 "self.x = x"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 使用 assertWarnsRegex 断言捕获 UserWarning 异常，并显示特定警告信息
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 将 M 类实例化并编译成 Torch Script
                torch.jit.script(M())

    @unittest.skipIf(
        sys.version_info[:2] < (3, 9), "Requires lowercase static typing (Python 3.9+)"
    )
    def test_annotated_empty_dict_lowercase(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置实例属性 self.x 为一个空的字典，键为字符串，值为整数（使用小写字典类型）
            def __init__(self):
                super().__init__()
                self.x: dict[str, int] = {}

            # 前向传播函数，接受参数 x，将其赋值给 self.x，并返回整数 1
            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        # 使用 assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常，并显示自定义消息和高亮的代码行 "self.x = x"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 使用 assertWarnsRegex 断言捕获 UserWarning 异常，并显示特定警告信息
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 将 M 类实例化并编译成 Torch Script
                torch.jit.script(M())

    def test_annotated_empty_optional(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置实例属性 self.x 为一个可选的字符串类型，默认为 None
            def __init__(self):
                super().__init__()
                self.x: Optional[str] = None

            # 前向传播函数，接受参数 x，将其赋值给 self.x，并返回整数 1
            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        # 使用 assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常，并显示自定义消息和高亮的代码行 "self.x = x"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            # 使用 assertWarnsRegex 断言捕获 UserWarning 异常，并显示特定警告信息
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 将 M 类实例化并编译成 Torch Script
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置实例属性 self.x 为一个空的列表，元素类型为整数，使用 torch.jit.annotate 进行类型注释
            def __init__(self):
                super().__init__()
                self.x = torch.jit.annotate(List[int], [])

            # 前向传播函数，接受参数 x，将其赋值给 self.x，并返回整数 1
            def forward(self, x: List[int]):
                self.x = x
                return 1

        # 使用 assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常，并显示自定义消息和高亮的代码行 "self.x = x"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 使用 assertWarnsRegex 断言捕获 UserWarning 异常，并显示特定警告信息
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 将 M 类实例化并编译成 Torch Script
                torch.jit.script(M())
    def test_annotated_with_jit_empty_list_lowercase(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类初始化方法
                super().__init__()
                # 使用torch.jit.annotate为self.x注释一个空列表(list[int])
                self.x = torch.jit.annotate(list[int], [])

            # 前向传播方法，参数x是一个int类型的列表
            def forward(self, x: list[int]):
                # 将输入的x赋值给self.x
                self.x = x
                # 返回常数1
                return 1

        # 断言捕获RuntimeError异常，且异常消息中包含"Tried to set nonexistent attribute"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 断言捕获UserWarning异常，且异常消息中包含指定内容
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 对类M进行脚本化编译
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类初始化方法
                super().__init__()
                # 使用torch.jit.annotate为self.x注释一个空字典(Dict[str, int])
                self.x = torch.jit.annotate(Dict[str, int], {})

            # 前向传播方法，参数x是一个字典，键为str，值为int
            def forward(self, x: Dict[str, int]):
                # 将输入的x赋值给self.x
                self.x = x
                # 返回常数1
                return 1

        # 断言捕获RuntimeError异常，且异常消息中包含"Tried to set nonexistent attribute"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 断言捕获UserWarning异常，且异常消息中包含指定内容
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 对类M进行脚本化编译
                torch.jit.script(M())

    @unittest.skipIf(
        sys.version_info[:2] < (3, 9), "Requires lowercase static typing (Python 3.9+)"
    )
    def test_annotated_with_jit_empty_dict_lowercase(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类初始化方法
                super().__init__()
                # 使用torch.jit.annotate为self.x注释一个空字典(dict[str, int])
                self.x = torch.jit.annotate(dict[str, int], {})

            # 前向传播方法，参数x是一个字典，键为str，值为int
            def forward(self, x: dict[str, int]):
                # 将输入的x赋值给self.x
                self.x = x
                # 返回常数1
                return 1

        # 断言捕获RuntimeError异常，且异常消息中包含"Tried to set nonexistent attribute"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            # 断言捕获UserWarning异常，且异常消息中包含指定内容
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 对类M进行脚本化编译
                torch.jit.script(M())

    def test_annotated_with_jit_empty_optional(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类初始化方法
                super().__init__()
                # 使用torch.jit.annotate为self.x注释一个空Optional类型(None)
                self.x = torch.jit.annotate(Optional[str], None)

            # 前向传播方法，参数x是一个Optional[str]类型
            def forward(self, x: Optional[str]):
                # 将输入的x赋值给self.x
                self.x = x
                # 返回常数1
                return 1

        # 断言捕获RuntimeError异常，且异常消息中包含"Wrong type for attribute assignment"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            # 断言捕获UserWarning异常，且异常消息中包含指定内容
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 对类M进行脚本化编译
                torch.jit.script(M())
    # 定义一个名为 test_annotated_with_torch_jit_import 的测试方法
    def test_annotated_with_torch_jit_import(self):
        # 导入 torch 中的 jit 模块
        from torch import jit
        
        # 定义一个名为 M 的继承自 torch.nn.Module 的类
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法
                super().__init__()
                # 使用 jit.annotate 方法为 self.x 创建一个类型注释，类型为 Optional[str]，初始值为 None
                self.x = jit.annotate(Optional[str], None)

            # 前向传播方法
            def forward(self, x: Optional[str]):
                # 将输入的 x 赋值给 self.x
                self.x = x
                # 返回固定值 1
                return 1

        # 使用 self.assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常，并期望异常信息包含 "Wrong type for attribute assignment"，高亮显示的代码片段是 "self.x = x"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            # 使用 self.assertWarnsRegex 断言捕获 UserWarning 警告，并期望警告信息包含 "doesn't support instance-level annotations on empty non-base types"
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support "
                "instance-level annotations on "
                "empty non-base types",
            ):
                # 使用 torch.jit.script 方法对 M 类进行脚本化，期望抛出异常和警告
                torch.jit.script(M())
```