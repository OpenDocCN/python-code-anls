# `.\pytorch\torchgen\selective_build\operator.py`

```py
# 引入未来版本的注解支持
from __future__ import annotations

# 引入数据类特性
from dataclasses import dataclass

# 表示一个选择性构建运算符的信息类，用于确定选择性/定制的PyTorch构建结果，
# 不包括所有支持运算符的注册代码。这样做是为了减小生成的二进制文件大小，
# 以便在对二进制大小要求严格的部署环境中使用。
#
@dataclass(frozen=True)
class SelectiveBuildOperator:
    # 运算符的名称，包括aten::等前缀
    # 运算符名称可能包含或不包含重载名称。如果该运算符名称不指定重载名称，
    # 则可以通过查看本类中 'include_all_overloads' 标志的值来确定
    # 此条目是指向具有该基本名称的运算符族，还是仅指向具有此名称的运算符。
    name: str

    # 如果这是一个根运算符（即直接从TorchScript模型等调用的运算符），
    # 则为True。只要该实例的pytorch库被构建的任何模型直接调用，
    # 此运算符就被视为根运算符。因此，在此pytorch库实例中使用的所有模型中，
    # 它可能不是根运算符。
    is_root_operator: bool

    # 这个运算符是否用于设备上的训练？如果为True，则需要使用这些信息生成
    # VariableType_N.cpp 中的代码，用于注册与训练相关的运算符。
    # 如果此运算符在此pytorch库实例中的一个或多个模型中用于训练，则此标志为True。
    is_used_for_training: bool

    # 如果为True，则表示此运算符实例（对象）指的是没有重载名称的运算符，
    # 应该适用于所有具有此运算符名称作为基本名称的重载。此标志仅适用于
    # 运算符名称中不包含点（.）字符的对象。
    #
    # 注意：此标志是当前静态选择性（定制）构建机制的临时解决方法，
    # 当确定是否选择运算符进行注册时，它基本上忽略重载名称。
    include_all_overloads: bool

    # 运算符级别的调试信息
    _debug_info: tuple[str, ...] | None

    @staticmethod
    def from_yaml_dict(
        op_name: str, op_info: dict[str, object]
    ) -> SelectiveBuildOperator:
        # 定义允许的键集合，用于验证传入参数的完整性
        allowed_keys = {
            "name",
            "is_root_operator",
            "is_used_for_training",
            "include_all_overloads",
            "debug_info",
        }

        # 检查传入参数中是否存在未被允许的顶级键
        if len(set(op_info.keys()) - allowed_keys) > 0:
            raise Exception(  # noqa: TRY002
                "Got unexpected top level keys: {}".format(
                    ",".join(set(op_info.keys()) - allowed_keys),
                )
            )

        # 如果传入参数包含 "name"，则验证其与当前操作符名称是否一致
        if "name" in op_info:
            assert op_name == op_info["name"]

        # 获取或设置操作符是否为根操作符，并确保其为布尔类型
        is_root_operator = op_info.get("is_root_operator", True)
        assert isinstance(is_root_operator, bool)

        # 获取或设置操作符是否用于训练，并确保其为布尔类型
        is_used_for_training = op_info.get("is_used_for_training", True)
        assert isinstance(is_used_for_training, bool)

        # 获取或设置是否包含所有重载版本，并确保其为布尔类型
        include_all_overloads = op_info.get("include_all_overloads", True)
        assert isinstance(include_all_overloads, bool)

        # 获取调试信息列表并转换为元组形式，用于内部使用
        debug_info: tuple[str, ...] | None = None
        if "debug_info" in op_info:
            di_list = op_info["debug_info"]
            assert isinstance(di_list, list)
            debug_info = tuple(str(x) for x in di_list)

        # 返回一个 SelectiveBuildOperator 对象，使用传入的参数
        return SelectiveBuildOperator(
            name=op_name,
            is_root_operator=is_root_operator,
            is_used_for_training=is_used_for_training,
            include_all_overloads=include_all_overloads,
            _debug_info=debug_info,
        )

    @staticmethod
    def from_legacy_operator_name_without_overload(
        name: str,
    ) -> SelectiveBuildOperator:
        # 创建一个 SelectiveBuildOperator 对象，从传入的操作符名称中创建，不包含重载
        return SelectiveBuildOperator(
            name=name,
            is_root_operator=True,
            is_used_for_training=True,
            include_all_overloads=True,
            _debug_info=None,
        )

    def to_dict(self) -> dict[str, object]:
        # 将对象的属性转换为字典形式
        ret: dict[str, object] = {
            "is_root_operator": self.is_root_operator,
            "is_used_for_training": self.is_used_for_training,
            "include_all_overloads": self.include_all_overloads,
        }
        # 如果存在调试信息，则添加到字典中
        if self._debug_info is not None:
            ret["debug_info"] = self._debug_info

        return ret
# 合并调试信息的函数，接受两个可选参数，分别是左右两个元组或None，返回合并后的元组或None
def merge_debug_info(
    lhs: tuple[str, ...] | None,
    rhs: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    # 如果左右参数都是None，则返回None
    if lhs is None and rhs is None:
        return None

    # 将左右参数合并成一个集合，确保每个条目只出现一次，然后转换成元组返回
    return tuple(set((lhs or ()) + (rhs or ())))


# 组合两个SelectiveBuildOperator对象的函数，返回一个新的SelectiveBuildOperator对象
def combine_operators(
    lhs: SelectiveBuildOperator, rhs: SelectiveBuildOperator
) -> SelectiveBuildOperator:
    # 如果左右对象的名称不同，抛出异常
    if str(lhs.name) != str(rhs.name):
        raise Exception(
            f"Expected both arguments to have the same name, but got '{str(lhs.name)}' and '{str(rhs.name)}' instead"
        )

    # 返回一个新的SelectiveBuildOperator对象，其中的is_root_operator、is_used_for_training、include_all_overloads属性取决于左右对象的对应属性，_debug_info属性调用merge_debug_info函数进行合并
    return SelectiveBuildOperator(
        name=lhs.name,
        is_root_operator=lhs.is_root_operator or rhs.is_root_operator,
        is_used_for_training=lhs.is_used_for_training or rhs.is_used_for_training,
        include_all_overloads=lhs.include_all_overloads or rhs.include_all_overloads,
        _debug_info=merge_debug_info(lhs._debug_info, rhs._debug_info),
    )


# 合并两个字典，每个字典的键是操作符名称，值是SelectiveBuildOperator对象
# 返回一个新的字典，包含合并后的操作符
def merge_operator_dicts(
    lhs: dict[str, SelectiveBuildOperator],
    rhs: dict[str, SelectiveBuildOperator],
) -> dict[str, SelectiveBuildOperator]:
    # 初始化一个空字典用于存放合并后的操作符
    operators: dict[str, SelectiveBuildOperator] = {}
    
    # 遍历左右两个字典的所有项，并合并同名操作符
    for op_name, op in list(lhs.items()) + list(rhs.items()):
        new_op = op
        # 如果操作符已经在结果字典中存在，则调用combine_operators函数合并同名操作符
        if op_name in operators:
            new_op = combine_operators(operators[op_name], op)

        # 更新结果字典中的操作符信息
        operators[op_name] = new_op

    # 返回合并后的操作符字典
    return operators


# 去除操作符名称中的重载部分，返回裁剪后的操作符名称
def strip_operator_overload_name(op_name: str) -> str:
    return op_name.split(".")[0]
```