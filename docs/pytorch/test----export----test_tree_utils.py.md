# `.\pytorch\test\export\test_tree_utils.py`

```
# Owner(s): ["oncall: export"]  # 指明代码的所有者信息

from collections import OrderedDict  # 导入 OrderedDict 类

import torch  # 导入 PyTorch 库
from torch._dynamo.test_case import TestCase  # 导入测试用例类

from torch.export._tree_utils import is_equivalent, reorder_kwargs  # 导入函数 is_equivalent 和 reorder_kwargs
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的函数
from torch.utils._pytree import tree_structure  # 导入函数 tree_structure


class TestTreeUtils(TestCase):
    def test_reorder_kwargs(self):
        original_kwargs = {"a": torch.tensor(0), "b": torch.tensor(1)}  # 原始关键字参数字典
        user_kwargs = {"b": torch.tensor(2), "a": torch.tensor(3)}  # 用户提供的关键字参数字典
        orig_spec = tree_structure(((), original_kwargs))  # 原始参数结构

        reordered_kwargs = reorder_kwargs(user_kwargs, orig_spec)  # 重新排序关键字参数

        # Key ordering should be the same
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0]),  # 断言关键字参数的顺序是否相同
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0]),  # 断言关键字参数的顺序是否相同

    def test_equivalence_check(self):
        tree1 = {"a": torch.tensor(0), "b": torch.tensor(1), "c": None}  # 第一个树结构
        tree2 = OrderedDict(a=torch.tensor(0), b=torch.tensor(1), c=None)  # 第二个树结构，使用 OrderedDict
        spec1 = tree_structure(tree1)  # 第一个树结构的结构描述
        spec2 = tree_structure(tree2)  # 第二个树结构的结构描述

        def dict_ordered_dict_eq(type1, context1, type2, context2):
            # 检查两个对象是否等价，考虑字典和 OrderedDict 的顺序
            if type1 is None or type2 is None:
                return type1 is type2 and context1 == context2

            if issubclass(type1, (dict, OrderedDict)) and issubclass(
                type2, (dict, OrderedDict)
            ):
                return context1 == context2

            return type1 is type2 and context1 == context2

        self.assertTrue(is_equivalent(spec1, spec2, dict_ordered_dict_eq))  # 断言两个树结构是否等价

        # Wrong ordering should still fail
        tree3 = OrderedDict(b=torch.tensor(1), a=torch.tensor(0))  # 错误顺序的 OrderedDict
        spec3 = tree_structure(tree3)  # 第三个树结构的结构描述
        self.assertFalse(is_equivalent(spec1, spec3, dict_ordered_dict_eq))  # 断言两个树结构是否不等价


if __name__ == "__main__":
    run_tests()  # 运行所有测试用例
```