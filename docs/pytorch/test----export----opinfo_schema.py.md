# `.\pytorch\test\export\opinfo_schema.py`

```
# Owner(s): ["oncall: export"]

import torch  # 导入PyTorch库
from torch._dispatch.python import enable_python_dispatcher  # 导入Python调度器相关功能
from torch._subclasses.schema_check_mode import SchemaCheckMode  # 导入模式检查相关类
from torch.fx.operator_schemas import normalize_function  # 导入规范化函数
from torch.testing._internal.common_device_type import (  # 导入设备类型测试相关功能
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db  # 导入操作数据库
from torch.testing._internal.common_utils import TestCase  # 导入测试用例类
from torch.utils._pytree import tree_map  # 导入树映射功能

# 简化C++类的命名
SchemaArgument = torch._C._SchemaArgument  # 定义模式参数类
SchemaArgType = torch._C._SchemaArgType  # 定义模式参数类型类
SchemaInfo = torch._C._SchemaInfo  # 定义模式信息类

test_classes = {}  # 初始化测试类字典


class PreDispatchSchemaCheckMode(SchemaCheckMode):
    """
    基于SchemaCheckMode构建的调度模式，检查预调度IR中的错误操作模式。
    旨在以即时模式运行具体输入中的操作，以查看它们是否错误地声明为函数式（别名或变异）。

    如果操作声称为函数式，并且检测到别名或变异，则会引发错误。
    如果模式允许别名或变异，则会静默错误 - 操作后续可能会分解并成为函数式。
    """

    def __init__(self):
        self._dispatch_key = torch._C.DispatchKey.PreDispatch  # 设置调度键为PreDispatch
        super().__init__()  # 调用父类构造函数

    def _may_alias_or_mutate(self, func, types, args, kwargs):
        def unwrap(e):
            if isinstance(e, torch.Tensor) and not type(e) == torch.Tensor:
                try:
                    return e.elem
                except AttributeError as t:
                    return e
            return e

        # 获取参数、输出
        schema_info = SchemaInfo(func._schema)  # 使用函数的模式信息初始化SchemaInfo
        pre_arguments = normalize_function(  # 规范化函数调用
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs
        schema_info.add_argument_values(pre_arguments)  # 添加预处理参数值
        out = func(*args, **kwargs)  # 调用函数
        tuple_out = out if isinstance(out, tuple) else (out,)  # 将输出转换为元组
        tuple_out = tree_map(unwrap, tuple_out)  # 对输出元组中的每个元素进行解包操作

        # 检查模式
        for i in range(len(func._schema.arguments)):
            for j in range(len(tuple_out)):
                if schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, j),
                    SchemaArgument(SchemaArgType.input, i),
                ):
                    return True
            if schema_info.is_mutable(
                SchemaArgument(SchemaArgType.input, i),
            ):
                return True

        return False

    # 仅为了访问错误操作而创建此方法
    # 重载 __torch_dispatch__ 方法，用于处理特定的函数调度
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            # 调用父类的 __torch_dispatch__ 方法处理函数调度
            return super().__torch_dispatch__(func, types, args=args, kwargs=kwargs)
        except RuntimeError as e:
            # 检查是否架构声明要么是别名或者变异
            alias_or_mutate = self._may_alias_or_mutate(func, types, args, kwargs)
            # 如果架构声明为别名或变异，将会进一步分解
            if not alias_or_mutate:
                # 获取异常消息
                msg = e.args[0]
                # 修改异常消息，指示操作 <func> 存在别名或变异，尽管声明为函数式操作
                e.args = (
                    f"""SchemaCheckMode failed with the following error on op <{func}>, meaning
    this op contains aliasing or mutations, despite claiming to be functional:\n\n"""
                    + msg,
                )
                # 抛出修改后的异常
                raise e
class TestOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float, torch.int))
    # 使用 ops 装饰器，指定操作的数据库和允许的数据类型
    def test_schema_check_op(self, device, dtype, op):
        # 获取操作的样本输入迭代器，使用指定的设备和数据类型，不需要梯度
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        # 获取迭代器中的下一个样本输入
        inputs = next(sample_inputs_itr)
        # 构建参数列表，包括输入和其他参数
        args = [inputs.input] + list(inputs.args)
        # 获取关键字参数
        kwargs = inputs.kwargs
        # 启用 Python 调度器
        with enable_python_dispatcher():
            # 进入预调度模式进行模式检查
            with PreDispatchSchemaCheckMode():
                # 执行操作的操作函数，传入参数和关键字参数
                op.op(*args, **kwargs)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # 运行测试
    run_tests()
```