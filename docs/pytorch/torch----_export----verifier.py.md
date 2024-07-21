# `.\pytorch\torch\_export\verifier.py`

```py
# mypy: allow-untyped-defs
# 引入inspect模块，用于获取对象信息
import inspect
# 引入math模块，提供数学函数
import math
# 引入operator模块，提供操作符函数
import operator
# 从collections.abc模块中导入Iterable抽象基类
from collections.abc import Iterable
# 从typing模块中导入类型相关的声明
from typing import Any, Dict, final, List, Optional, Tuple, Type

# 引入torch模块，主要用于深度学习任务
import torch
# 从torch._ops中导入HigherOrderOperator和OpOverload类
from torch._ops import HigherOrderOperator, OpOverload
# 从torch._subclasses.fake_tensor中导入FakeTensor类
from torch._subclasses.fake_tensor import FakeTensor
# 从torch.export.exported_program中导入ExportedProgram类
from torch.export.exported_program import ExportedProgram
# 从torch.export.graph_signature中导入各种参数类
from torch.export.graph_signature import (
    CustomObjArgument,
    InputKind,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)
# 从torch.fx中导入GraphModule类
from torch.fx import GraphModule
# 从torch.fx.experimental.symbolic_shapes中导入符号化类型类
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt


# 定义一个异常类SpecViolationError，用于规范违例
class SpecViolationError(Exception):
    pass


# 定义函数is_functional，判断操作是否为functional
def is_functional(op: OpOverload) -> bool:
    # 判断op对应的模式是否为不可变的
    return not op._schema.is_mutable


# 定义内部函数_check_has_fake_tensor，用于检查节点是否包含FakeTensor
def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    # TODO(angelayi): remove this in favor of _check_val
    # 调用_check_val函数检查节点
    return _check_val(node)


# 定义内部函数_check_val，用于检查节点的值是否符合规范
def _check_val(node: torch.fx.Node) -> None:
    # 定义内部函数_check_correct_val，用于检查值是否正确
    def _check_correct_val(val):
        if val is None:
            return True
        elif isinstance(val, (int, bool, str, float)):
            return True
        elif isinstance(val, (torch.memory_format, torch.dtype, torch.device, torch.layout)):
            return True
        elif isinstance(val, (FakeTensor, torch.Tensor)):  # TODO(zhxchen17) Remove Tensor.
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, CustomObjArgument):
            return True
        elif isinstance(val, Iterable):
            return all(_check_correct_val(x) for x in val)
        return False

    # 定义内部函数_no_returns，检查操作是否没有返回值
    def _no_returns(op):
        if not isinstance(op, OpOverload):
            return False
        return len(op._schema.returns) == 0

    # 如果节点的元数据中不包含'val'字段，则抛出异常
    if "val" not in node.meta:
        if node.op == "call_function" and _no_returns(node.target):
            return
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")

    # 获取节点的'val'值
    val = node.meta["val"]
    # 检查'val'值是否符合规范
    if not _check_correct_val(val):
        raise SpecViolationError(f"Node.meta {node.name} has invalid val field {val}")


# 定义内部函数_check_torch_fn，用于检查节点是否包含正确的torch_fn元数据
def _check_torch_fn(node: torch.fx.Node) -> None:
    # 获取节点的torch_fn元数据
    torch_fn = node.meta.get("torch_fn")
    # 如果torch_fn为None，则抛出异常
    if torch_fn is None:
        raise SpecViolationError(f"Unable to find torch_fn metadata for node {node.name}")
    # 检查torch_fn是否为元组，并且元组的第一个和第二个元素都是字符串
    if (
        not isinstance(torch_fn, tuple) and
        isinstance(torch_fn[0], str) and
        isinstance(torch_fn[1], str)
    ):
        raise SpecViolationError(f"Node.meta {node.name} has invalid torch_fn field {torch_fn}")


# 定义_VerifierMeta元类
class _VerifierMeta(type):
    # _registry属性，用于存储Verifier类的注册信息
    _registry: Dict[str, Type['Verifier']] = {}
    # 定义一个特殊的方法 __new__，用于创建类的实例
    def __new__(metacls, name, bases, attrs):
        # 检查是否存在父类（即判断是否为子类）
        if bases:
            # 如果子类中定义了 "check" 属性或 "_check_graph_module" 方法，则抛出语法错误异常
            if "check" in attrs or "_check_graph_module" in attrs:
                raise SyntaxError("Overriding method check is not allowed.")
            # 断言确保子类中存在名为 "dialect" 的属性，并且其值不为 "ATEN"
            assert "dialect" in attrs and attrs["dialect"] != "ATEN"
        else:
            # 如果是基类，则要求必须存在 "check" 属性
            assert "check" in attrs
            # 断言确保基类中必须存在 "_check_graph_module" 方法
            assert "_check_graph_module" in attrs
            # 断言确保基类中的 "dialect" 属性必须为 "ATEN"
            assert attrs["dialect"] == "ATEN"

        # 断言确保 "dialect" 属性是一个字符串类型
        assert isinstance(attrs["dialect"], str)
        
        # 创建类实例
        ret = type.__new__(metacls, name, bases, attrs)
        # 将创建的类实例与其 "dialect" 属性关联存储到 _registry 字典中
        metacls._registry[attrs["dialect"]] = ret  # type: ignore[assignment]
        
        # 返回创建的类实例
        return ret
def getattr_recursive(obj: Any, target: str) -> Any:
    # 将目标字符串按 '.' 分割成列表，表示访问对象的层级关系
    target_atoms = target.split('.')
    # 初始化属性迭代器为目标对象本身
    attr_itr = obj
    # 遍历目标字符串中的每个层级
    for i, atom in enumerate(target_atoms):
        # 检查当前迭代器是否具有指定属性，否则抛出运行时错误
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        # 更新迭代器为当前层级的属性值
        attr_itr = getattr(attr_itr, atom)
    # 返回最终获取到的属性值
    return attr_itr


class Verifier(metaclass=_VerifierMeta):
    # 定义方言为 "ATEN"
    dialect = "ATEN"

    def allowed_builtin_ops(self) -> List:
        # 返回允许的内置运算符列表
        return [
            operator.getitem,
            operator.add,
            operator.mul,
            operator.sub,
            operator.truediv,
            operator.ge,
            operator.le,
            operator.gt,
            operator.lt,
            operator.eq,
            operator.ne,
            operator.floordiv,
            operator.mod,
            operator.and_,
            operator.or_,
            operator.not_,
            operator.pow,
            operator.neg,
            operator.abs,
            math.ceil,
            math.floor,
        ]

    def allowed_op_types(self) -> Tuple[Type[Any], ...]:
        # 返回允许的操作类型元组，包括 OpOverload、HigherOrderOperator 和特定注册的操作类型
        from torch._export.serde.serialize import allowed_registered_op_types  # 避免循环导入
        return (OpOverload, HigherOrderOperator, *allowed_registered_op_types())

    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        # 返回允许进行 getattr 操作的类型元组，仅包含 torch.fx.GraphModule
        return (torch.fx.GraphModule,)

    def check_valid_op(self, op):
        # 检查操作的有效性，具体实现未提供
        pass

    def check_additional(self, gm: GraphModule) -> None:
        """
        特定于某些方言的额外检查。
        """
        # 针对给定的图模块执行特定方言的额外检查，但未提供具体实现
        pass

    @final
    def check(self, ep: ExportedProgram) -> None:
        # 检查导出程序的结构
        self._check_graph_module(ep.graph_module)
        _verify_exported_program_signature(ep)

    @final
class TrainingIRVerifier(Verifier):
    # 定义方言为 "TRAINING"
    dialect = "TRAINING"


def _verify_exported_program_signature(exported_program) -> None:
    # 检查导出程序的签名是否匹配
    gs = exported_program.graph_signature

    # 检查图中每个节点在签名中是否存在
    input_node_names = [node.name for node in exported_program.graph.nodes if node.op == "placeholder"]

    if len(input_node_names) != len(gs.input_specs):
        # 如果图输入节点数量与签名中指定输入数量不匹配，则引发规范违规错误
        raise SpecViolationError(
            f"Number of graph inputs ({len(input_node_names)}) "
            f"does not match number of inputs in the graph signature ({len(gs.user_inputs)})"
        )

    # 检查输出
    output_node = list(exported_program.graph.nodes)[-1]
    assert output_node.op == "output"
    output_nodes = [
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in output_node.args[0]
    ]
    # 检查输出节点的数量是否与图形签名中指定的输出数量相匹配
    if len(output_nodes) != len(gs.output_specs):
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            "Than the number of outputs specified by the graph signature: \n"
            f"Number of mutated buffers: {len(gs.buffers_to_mutate)}. \n"
            f"Number of user outputs: {len(gs.user_outputs)}. \n"
        )

    # 计算可变节点的范围和用户输出节点的范围
    num_tokens = len(gs.output_tokens)
    end = len(gs.buffers_to_mutate) + len(gs.user_inputs_to_mutate) + num_tokens
    mutate_nodes: List[str] = output_nodes[num_tokens:end]
    user_output_nodes = output_nodes[end:end + len(gs.user_outputs)]

    # 检查每个可变节点是否有效
    for mutation_node in mutate_nodes:
        if mutation_node in gs.buffers_to_mutate:
            # 如果可变节点是缓冲区，确保其指向一个存在的缓冲区
            if gs.buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(
                    f"Buffer output {mutation_node} does not point to a buffer that exists. \n"
                    f"Dict of buffers that are mutated, in order: {gs.buffers_to_mutate} \n"
                    f"Buffer nodes available: {gs.buffers} \n"
                )
        elif mutation_node in gs.user_inputs_to_mutate:
            # 如果可变节点是用户输入，确保其指向一个存在的用户输入
            if gs.user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(
                    f"User input output {mutation_node} does not point to a user input that exists. \n"
                    f"Dict of user inputs that are mutated, in order: {gs.user_inputs_to_mutate} \n"
                    f"User input nodes available: {gs.user_inputs} \n")
        else:
            # 如果可变节点既不是缓冲区也不是用户输入，抛出规范违规错误
            raise SpecViolationError(
                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
                f"Buffers to mutate: {gs.buffers_to_mutate}, User inputs to mutate: {gs.user_inputs_to_mutate}"
            )

    # 检查用户输出节点的顺序和正确性
    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_outputs}. "
            )
# 定义函数 load_verifier，接受一个字符串参数 dialect 并返回一个 Verifier 类型或者 None
def load_verifier(dialect: str) -> Optional[Type[Verifier]]:
    # 如果 dialect 是 "ATEN" 或者空字符串 ""
    if dialect == "ATEN" or dialect == "":
        # 从 _VerifierMeta._registry 中获取键为 dialect 的值，并返回
        return _VerifierMeta._registry.get(dialect)
    # 如果 dialect 不是 "ATEN" 或者空字符串 ""
    else:
        # 直接从 _VerifierMeta._registry 中获取键为 dialect 的值，并返回
        return _VerifierMeta._registry[dialect]
```