# `.\pytorch\tools\test\test_executorch_unboxing.py`

```
import unittest  # 导入unittest模块，用于编写和运行测试

from types import ModuleType  # 导入ModuleType类，用于创建新的模块类型

from torchgen import local  # 导入torchgen中的local模块
from torchgen.api import cpp as aten_cpp, types as aten_types  # 导入torchgen.api中的cpp和types模块
from torchgen.api.types import (  # 导入torchgen.api.types中的多个类和函数
    ArgName,
    BaseCType,
    ConstRefCType,
    MutRefCType,
    NamedCType,
)
from torchgen.executorch.api import et_cpp as et_cpp, types as et_types  # 导入torchgen.executorch.api中的et_cpp和types模块
from torchgen.executorch.api.unboxing import Unboxing  # 导入torchgen.executorch.api.unboxing中的Unboxing类
from torchgen.model import BaseTy, BaseType, ListType, OptionalType, Type  # 导入torchgen.model中的多个类

# 定义函数，用于包装aten的参数类型，返回一个NamedCType对象
def aten_argumenttype_type_wrapper(
    t: Type, *, mutable: bool, binds: ArgName, remove_non_owning_ref_types: bool = False
) -> NamedCType:
    return aten_cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )

# 创建ATEN_UNBOXING对象，使用Unboxing类初始化，指定argument_type_gen参数为aten_argumenttype_type_wrapper函数
ATEN_UNBOXING = Unboxing(argument_type_gen=aten_argumenttype_type_wrapper)

# 创建ET_UNBOXING对象，使用Unboxing类初始化，指定argument_type_gen参数为et_cpp.argumenttype_type
ET_UNBOXING = Unboxing(argument_type_gen=et_cpp.argumenttype_type)

# 定义测试类TestUnboxing，继承自unittest.TestCase
class TestUnboxing(unittest.TestCase):
    """
    Could use torch.testing._internal.common_utils to reduce boilerplate.
    GH CI job doesn't build torch before running tools unit tests, hence
    manually adding these parametrized tests.
    """

    # 使用local.parametrize装饰器参数化测试方法
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义测试方法test_symint_argument_translate_ctype_aten，返回None
    def test_symint_argument_translate_ctype_aten(self) -> None:
        # 测试是否能正确将`SymInt[]` JIT参数转换为C++参数
        # 由于Executorch不使用symint签名，应该是`IntArrayRef`

        # pyre-fixme[16]: `enum.Enum` has no attribute `SymInt`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)

        # 调用ATEN_UNBOXING对象的argumenttype_evalue_convert方法，获取返回值
        out_name, ctype, _, _ = ATEN_UNBOXING.argumenttype_evalue_convert(
            t=symint_list_type, arg_name="size", mutable=False
        )

        # 断言out_name为"size_list_out"
        self.assertEqual(out_name, "size_list_out")
        # 断言ctype的类型为BaseCType
        self.assertIsInstance(ctype, BaseCType)
        # pyre-fixme[16]:
        # 断言ctype等于aten_types.intArrayRefT
        self.assertEqual(ctype, aten_types.BaseCType(aten_types.intArrayRefT))

    # 使用local.parametrize装饰器参数化测试方法
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_symint_argument_translate_ctype_executorch(self) -> None:
        # 测试 `SymInt[]` JIT 参数能否正确翻译为 C++ 参数。
        # 由于 Executorch 不使用 symint sig，应该是 `IntArrayRef` 类型。

        # pyre-fixme[16]: `enum.Enum` has no attribute `SymInt`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        # 定义 symint_list_type 变量，其类型为 ListType，元素类型为 BaseType(BaseTy.SymInt)，大小为 None
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)

        # 调用 ET_UNBOXING 的 argumenttype_evalue_convert 方法，转换 symint_list_type 类型的参数
        out_name, ctype, _, _ = ET_UNBOXING.argumenttype_evalue_convert(
            t=symint_list_type, arg_name="size", mutable=False
        )

        # 断言 out_name 的值应为 "size_list_out"
        self.assertEqual(out_name, "size_list_out")
        # 断言 ctype 的类型应为 et_types.ArrayRefCType
        self.assertIsInstance(ctype, et_types.ArrayRefCType)
        # pyre-fixme[16]:
        # 断言 ctype 应为 et_types.ArrayRefCType，其元素类型为 BaseCType(aten_types.longT)
        self.assertEqual(
            ctype, et_types.ArrayRefCType(elem=BaseCType(aten_types.longT))
        )

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_const_tensor_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        # 定义 tensor_type 变量，其类型为 BaseType(BaseTy.Tensor)
        tensor_type = BaseType(BaseTy.Tensor)

        # 调用 unboxing 的 argumenttype_evalue_convert 方法，转换 tensor_type 类型的参数
        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_type, arg_name="self", mutable=False
        )

        # 断言 out_name 的值应为 "self_base"
        self.assertEqual(out_name, "self_base")
        # pyre-fixme[16]:
        # 断言 ctype 应为 ConstRefCType，其内部类型为 BaseCType(types.tensorT)
        self.assertEqual(ctype, ConstRefCType(BaseCType(types.tensorT)))

    def test_const_tensor_argument_translate_ctype_aten(self) -> None:
        # 调用 _test_const_tensor_argument_translate_ctype 方法，使用 ATEN_UNBOXING 和 aten_types 参数
        self._test_const_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_const_tensor_argument_translate_ctype_executorch(self) -> None:
        # 调用 _test_const_tensor_argument_translate_ctype 方法，使用 ET_UNBOXING 和 et_types 参数
        self._test_const_tensor_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_mutable_tensor_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        # 定义 tensor_type 变量，其类型为 BaseType(BaseTy.Tensor)
        tensor_type = BaseType(BaseTy.Tensor)

        # 调用 unboxing 的 argumenttype_evalue_convert 方法，转换 tensor_type 类型的可变参数
        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_type, arg_name="out", mutable=True
        )

        # 断言 out_name 的值应为 "out_base"
        self.assertEqual(out_name, "out_base")
        # pyre-fixme[16]:
        # 断言 ctype 应为 MutRefCType，其内部类型为 BaseCType(types.tensorT)
        self.assertEqual(ctype, MutRefCType(BaseCType(types.tensorT)))

    def test_mutable_tensor_argument_translate_ctype_aten(self) -> None:
        # 调用 _test_mutable_tensor_argument_translate_ctype 方法，使用 ATEN_UNBOXING 和 aten_types 参数
        self._test_mutable_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_mutable_tensor_argument_translate_ctype_executorch(self) -> None:
        # 调用 _test_mutable_tensor_argument_translate_ctype 方法，使用 ET_UNBOXING 和 et_types 参数
        self._test_mutable_tensor_argument_translate_ctype(ET_UNBOXING, et_types)
    # 使用 @local.parametrize 装饰器设置参数化测试的参数，禁用对可变张量使用常量引用和使用列表引用的选项
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义私有方法 _test_tensor_list_argument_translate_ctype，用于测试张量列表参数的类型转换
    def _test_tensor_list_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        # 定义一个列表类型变量 tensor_list_type，元素类型为 BaseType(BaseTy.Tensor)，大小为 None
        tensor_list_type = ListType(elem=BaseType(BaseTy.Tensor), size=None)

        # 调用 unboxing 对象的 argumenttype_evalue_convert 方法，将 tensor_list_type 转换为 ctype
        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_list_type, arg_name="out", mutable=True
        )

        # 断言输出的名称应为 "out_list_out"
        self.assertEqual(out_name, "out_list_out")
        # pyre-fixme[16]: 
        # 断言 ctype 应为 types.tensorListT 对应的 BaseCType
        self.assertEqual(ctype, BaseCType(types.tensorListT))

    # 定义测试方法 test_tensor_list_argument_translate_ctype_aten，测试 ATEN 下的张量列表参数类型转换
    def test_tensor_list_argument_translate_ctype_aten(self) -> None:
        self._test_tensor_list_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    # 定义测试方法 test_tensor_list_argument_translate_ctype_executorch，测试 ET 下的张量列表参数类型转换
    def test_tensor_list_argument_translate_ctype_executorch(self) -> None:
        self._test_tensor_list_argument_translate_ctype(ET_UNBOXING, et_types)

    # 使用 @local.parametrize 装饰器设置参数化测试的参数，禁用对可变张量使用常量引用和使用列表引用的选项
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义私有方法 _test_optional_int_argument_translate_ctype，用于测试可选整数参数的类型转换
    def _test_optional_int_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        # 定义一个可选类型变量 optional_int_type，元素类型为 BaseType(BaseTy.int)
        optional_int_type = OptionalType(elem=BaseType(BaseTy.int))

        # 调用 unboxing 对象的 argumenttype_evalue_convert 方法，将 optional_int_type 转换为 ctype
        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=optional_int_type, arg_name="something", mutable=True
        )

        # 断言输出的名称应为 "something_opt_out"
        self.assertEqual(out_name, "something_opt_out")
        # pyre-fixme[16]: 
        # 断言 ctype 应为 types.OptionalCType(types.longT) 的结果
        self.assertEqual(ctype, types.OptionalCType(BaseCType(types.longT)))

    # 定义测试方法 test_optional_int_argument_translate_ctype_aten，测试 ATEN 下的可选整数参数类型转换
    def test_optional_int_argument_translate_ctype_aten(self) -> None:
        self._test_optional_int_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    # 定义测试方法 test_optional_int_argument_translate_ctype_executorch，测试 ET 下的可选整数参数类型转换
    def test_optional_int_argument_translate_ctype_executorch(self) -> None:
        self._test_optional_int_argument_translate_ctype(ET_UNBOXING, et_types)
```