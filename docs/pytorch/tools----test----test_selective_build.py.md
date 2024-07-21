# `.\pytorch\tools\test\test_selective_build.py`

```py
import unittest  # 导入unittest模块，用于编写和运行测试用例

from torchgen.model import Location, NativeFunction  # 导入torchgen库中的Location和NativeFunction类
from torchgen.selective_build.operator import *  # 导入torchgen.selective_build.operator库的所有内容（除了F403号警告）
from torchgen.selective_build.selector import (  # 导入torchgen.selective_build.selector库中的以下函数和类
    combine_selective_builders,
    SelectiveBuilder,
)


class TestSelectiveBuild(unittest.TestCase):  # 定义测试类TestSelectiveBuild，继承自unittest.TestCase

    def test_selective_build_operator(self) -> None:  # 定义测试方法test_selective_build_operator，返回类型为None
        op = SelectiveBuildOperator(  # 创建SelectiveBuildOperator对象op，用于特定构建操作
            "aten::add.int",  # 操作的名称
            is_root_operator=True,  # 是否为根操作符
            is_used_for_training=False,  # 是否用于训练
            include_all_overloads=False,  # 是否包括所有重载
            _debug_info=None,  # 调试信息（未使用）
        )
        self.assertTrue(op.is_root_operator)  # 断言：op应为根操作符
        self.assertFalse(op.is_used_for_training)  # 断言：op不应用于训练
        self.assertFalse(op.include_all_overloads)  # 断言：op不应包括所有重载

    def test_selector_factory(self) -> None:  # 定义测试方法test_selector_factory，返回类型为None
        yaml_config_v1 = """
debug_info:
  - model1@v100
  - model2@v51
operators:
  aten::add:
    is_used_for_training: No
    is_root_operator: Yes
    include_all_overloads: Yes
  aten::add.int:
    is_used_for_training: Yes
    is_root_operator: No
    include_all_overloads: No
  aten::mul.int:
    is_used_for_training: Yes
    is_root_operator: No
    include_all_overloads: No
"""

        yaml_config_v2 = """
debug_info:
  - model1@v100
  - model2@v51
operators:
  aten::sub:
    is_used_for_training: No
    is_root_operator: Yes
    include_all_overloads: No
    debug_info:
      - model1@v100
  aten::sub.int:
    is_used_for_training: Yes
    is_root_operator: No
    include_all_overloads: No
"""

        # 创建测试方法test_operator_combine，用于组合SelectiveBuildOperator对象
        def test_operator_combine(self) -> None:
            op1 = SelectiveBuildOperator(  # 创建SelectiveBuildOperator对象op1
                "aten::add.int",  # 操作的名称
                is_root_operator=True,  # 是否为根操作符
                is_used_for_training=False,  # 是否用于训练
                include_all_overloads=False,  # 是否包括所有重载
                _debug_info=None,  # 调试信息（未使用）
            )
            op2 = SelectiveBuildOperator(  # 创建SelectiveBuildOperator对象op2
                "aten::add.int",  # 操作的名称
                is_root_operator=False,  # 是否为根操作符
                is_used_for_training=False,  # 是否用于训练
                include_all_overloads=False,  # 是否包括所有重载
                _debug_info=None,  # 调试信息（未使用）
            )
            op3 = SelectiveBuildOperator(  # 创建SelectiveBuildOperator对象op3
                "aten::add",  # 操作的名称
                is_root_operator=True,  # 是否为根操作符
                is_used_for_training=False,  # 是否用于训练
                include_all_overloads=False,  # 是否包括所有重载
                _debug_info=None,  # 调试信息（未使用）
            )
            op4 = SelectiveBuildOperator(  # 创建SelectiveBuildOperator对象op4
                "aten::add.int",  # 操作的名称
                is_root_operator=True,  # 是否为根操作符
                is_used_for_training=True,  # 是否用于训练
                include_all_overloads=False,  # 是否包括所有重载
                _debug_info=None,  # 调试信息（未使用）
            )

            op5 = combine_operators(op1, op2)  # 调用combine_operators函数组合op1和op2，返回新的SelectiveBuildOperator对象op5

            self.assertTrue(op5.is_root_operator)  # 断言：op5应为根操作符
            self.assertFalse(op5.is_used_for_training)  # 断言：op5不应用于训练

            op6 = combine_operators(op1, op4)  # 调用combine_operators函数组合op1和op4，返回新的SelectiveBuildOperator对象op6

            self.assertTrue(op6.is_root_operator)  # 断言：op6应为根操作符
            self.assertTrue(op6.is_used_for_training)  # 断言：op6应用于训练

            def gen_new_op():  # 定义函数gen_new_op，用于生成新的操作组合
                return combine_operators(op1, op3)  # 调用combine_operators函数组合op1和op3

            self.assertRaises(Exception, gen_new_op)  # 断言：调用gen_new_op函数应引发异常

        def test_training_op_fetch(self) -> None:  # 定义测试方法test_training_op_fetch，返回类型为None
            yaml_config = """
operators:
  aten::add.int:
    is_used_for_training: No
    is_root_operator: Yes
    include_all_overloads: No
  aten::add:
"""
    # 是否用于训练：是
    is_used_for_training: Yes
    # 是否为根操作符：否
    is_root_operator: No
    # 包括所有重载版本：是
    include_all_overloads: Yes
    # 从 YAML 字符串创建 SelectiveBuilder 对象
    selector = SelectiveBuilder.from_yaml_str(yaml_config)
    # 断言操作符 "aten::add.int" 被选择用于训练
    self.assertTrue(selector.is_operator_selected_for_training("aten::add.int"))
    # 断言操作符 "aten::add" 被选择用于训练
    self.assertTrue(selector.is_operator_selected_for_training("aten::add"))

def test_kernel_dtypes(self) -> None:
    # 定义包含 kernel 元数据的 YAML 配置字符串
    yaml_config = """
kernel_metadata:
  add_kernel:
    - int8
    - int32
  sub_kernel:
    - int16
    - int32
  add/sub_kernel:
    - float
    - complex
"""
    # 从 YAML 字符串创建 SelectiveBuilder 对象
    selector = SelectiveBuilder.from_yaml_str(yaml_config)
    # 断言 "add_kernel" 的 "int32" 数据类型被选择
    self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int32"))
    # 断言 "add_kernel" 的 "int8" 数据类型被选择
    self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int8"))
    # 断言 "add_kernel" 的 "int16" 数据类型未被选择
    self.assertFalse(selector.is_kernel_dtype_selected("add_kernel", "int16"))
    # 断言 "add1_kernel" 的 "int32" 数据类型未被选择
    self.assertFalse(selector.is_kernel_dtype_selected("add1_kernel", "int32"))
    # 断言 "add_kernel" 的 "float" 数据类型未被选择
    self.assertFalse(selector.is_kernel_dtype_selected("add_kernel", "float"))

    # 断言 "add/sub_kernel" 的 "float" 数据类型被选择
    self.assertTrue(selector.is_kernel_dtype_selected("add/sub_kernel", "float"))
    # 断言 "add/sub_kernel" 的 "complex" 数据类型被选择
    self.assertTrue(selector.is_kernel_dtype_selected("add/sub_kernel", "complex"))
    # 断言 "add/sub_kernel" 的 "int16" 数据类型未被选择
    self.assertFalse(selector.is_kernel_dtype_selected("add/sub_kernel", "int16"))
    # 断言 "add/sub_kernel" 的 "int32" 数据类型未被选择
    self.assertFalse(selector.is_kernel_dtype_selected("add/sub_kernel", "int32"))

def test_merge_kernel_dtypes(self) -> None:
    # 定义两个包含 kernel 元数据的 YAML 配置字符串
    yaml_config1 = """
kernel_metadata:
  add_kernel:
    - int8
  add/sub_kernel:
    - float
    - complex
    - none
  mul_kernel:
    - int8
"""

    yaml_config2 = """
kernel_metadata:
  add_kernel:
    - int32
  sub_kernel:
    - int16
    - int32
  add/sub_kernel:
    - float
    - complex
"""

    # 从两个 YAML 字符串分别创建 SelectiveBuilder 对象
    selector1 = SelectiveBuilder.from_yaml_str(yaml_config1)
    selector2 = SelectiveBuilder.from_yaml_str(yaml_config2)

    # 合并两个 SelectiveBuilder 对象
    selector = combine_selective_builders(selector1, selector2)

    # 断言合并后的结果
    self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int32"))
    self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int8"))
    self.assertFalse(selector.is_kernel_dtype_selected("add_kernel", "int16"))
    self.assertFalse(selector.is_kernel_dtype_selected("add1_kernel", "int32"))
    self.assertFalse(selector.is_kernel_dtype_selected("add_kernel", "float"))

    self.assertTrue(selector.is_kernel_dtype_selected("add/sub_kernel", "float"))
    self.assertTrue(selector.is_kernel_dtype_selected("add/sub_kernel", "complex"))
    self.assertTrue(selector.is_kernel_dtype_selected("add/sub_kernel", "none"))
    self.assertFalse(selector.is_kernel_dtype_selected("add/sub_kernel", "int16"))
    self.assertFalse(selector.is_kernel_dtype_selected("add/sub_kernel", "int32"))

    self.assertTrue(selector.is_kernel_dtype_selected("mul_kernel", "int8"))
    self.assertFalse(selector.is_kernel_dtype_selected("mul_kernel", "int32"))

def test_all_kernel_dtypes_selected(self) -> None:
    # 定义包含 include_all_non_op_selectives 的 YAML 配置字符串
    yaml_config = """
include_all_non_op_selectives: True
"""
# 创建一个 SelectiveBuilder 对象，使用给定的 YAML 字符串配置
selector = SelectiveBuilder.from_yaml_str(yaml_config)

# 断言指定的内核（"add_kernel"）对于指定的数据类型（"int32"）被正确选择
self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int32"))
# 断言指定的内核（"add_kernel"）对于指定的数据类型（"int8"）被正确选择
self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int8"))
# 断言指定的内核（"add_kernel"）对于指定的数据类型（"int16"）被正确选择
self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "int16"))
# 断言指定的内核（"add1_kernel"）对于指定的数据类型（"int32"）被正确选择
self.assertTrue(selector.is_kernel_dtype_selected("add1_kernel", "int32"))
# 断言指定的内核（"add_kernel"）对于指定的数据类型（"float"）被正确选择
self.assertTrue(selector.is_kernel_dtype_selected("add_kernel", "float"))

# 创建一个 SelectiveBuilder 对象，使用给定的 YAML 字符串配置
selector = SelectiveBuilder.from_yaml_str(yaml_config)

# 从 YAML 中创建一个 NativeFunction 对象
native_function, _ = NativeFunction.from_yaml(
    {"func": "custom::add() -> Tensor"},
    loc=Location(__file__, 1),
    valid_tags=set(),
)
# 断言该自定义函数被正确选择
self.assertTrue(selector.is_native_function_selected(native_function))

# 创建一个 SelectiveBuilder 对象，使用给定的 YAML 字符串配置
selector = SelectiveBuilder.from_yaml_str(yaml_config)

# 断言对于 "aten::add.out" 内核，返回与预期相同的已选择内核列表
self.assertListEqual(
    ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
    selector.et_get_selected_kernels(
        "aten::add.out",
        [
            "v1/6;0,1|6;0,1|6;0,1|6;0,1",
            "v1/3;0,1|3;0,1|3;0,1|3;0,1",
            "v1/6;1,0|6;0,1|6;0,1|6;0,1",
        ],
    ),
)
# 断言对于 "aten::sub.out" 内核，返回与预期相同的已选择内核列表
self.assertListEqual(
    ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
    selector.et_get_selected_kernels(
        "aten::sub.out", ["v1/6;0,1|6;0,1|6;0,1|6;0,1"]
    ),
)
# 断言对于 "aten::mul.out" 内核，返回空的已选择内核列表
self.assertListEqual(
    [],
    selector.et_get_selected_kernels(
        "aten::mul.out", ["v1/6;0,1|6;0,1|6;0,1|6;0,1"]
    ),
)
# 断言对于 "aten::add.out" 内核，返回与预期相同的已选择内核列表
self.assertListEqual(
    ["v2/6;0,1|6;0,1|6;0,1|6;0,1"],
    selector.et_get_selected_kernels(
        "aten::add.out", ["v2/6;0,1|6;0,1|6;0,1|6;0,1"]
    ),
)
```