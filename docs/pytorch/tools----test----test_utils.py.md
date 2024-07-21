# `.\pytorch\tools\test\test_utils.py`

```py
import unittest  # 导入单元测试模块

from torchgen.utils import NamespaceHelper  # 从torchgen.utils模块导入NamespaceHelper类


class TestNamespaceHelper(unittest.TestCase):  # 定义测试类TestNamespaceHelper，继承unittest.TestCase

    def test_create_from_namespaced_tuple(self) -> None:
        # 测试从命名空间实体创建NamespaceHelper对象
        helper = NamespaceHelper.from_namespaced_entity("aten::add")
        self.assertEqual(helper.entity_name, "add")  # 断言实体名称为"add"
        self.assertEqual(helper.get_cpp_namespace(), "aten")  # 断言获取的C++命名空间为"aten"

    def test_default_namespace(self) -> None:
        # 测试默认命名空间处理
        helper = NamespaceHelper.from_namespaced_entity("add")
        self.assertEqual(helper.entity_name, "add")  # 断言实体名称为"add"
        self.assertEqual(helper.get_cpp_namespace(), "")  # 断言获取的C++命名空间为空字符串
        self.assertEqual(helper.get_cpp_namespace("default"), "default")  # 断言获取的C++命名空间为"default"

    def test_namespace_levels_more_than_max(self) -> None:
        # 测试超过最大命名空间级别时引发断言错误
        with self.assertRaises(AssertionError):
            NamespaceHelper(
                namespace_str="custom_1::custom_2", entity_name="", max_level=1
            )
```