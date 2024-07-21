# `.\pytorch\torch\fx\experimental\validator.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和库
import functools  # 导入 functools 模块，提供高阶函数和操作工具
import logging    # 导入 logging 模块，用于日志记录
import math       # 导入 math 模块，提供数学函数
import operator   # 导入 operator 模块，提供操作符的函数接口
import sympy      # 导入 sympy 模块，用于符号计算
import builtins   # 导入 builtins 模块，包含内建函数的标准集合

from dataclasses import dataclass   # 导入 dataclass 装饰器，用于创建不可变数据对象
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union   # 导入类型提示模块中的各种类型

import torch       # 导入 PyTorch 深度学习框架
import torch.fx    # 导入 PyTorch FX 模块，用于对 PyTorch 模型进行分析和转换
import torch.fx.traceback as fx_traceback   # 导入 PyTorch FX 的 traceback 模块，用于追踪调试信息

from torch._dynamo.exc import TorchDynamoException   # 从 Torch._dynamo.exc 模块导入 TorchDynamoException 异常类
from torch.fx.node import Argument, Target   # 从 torch.fx.node 模块导入 Argument 和 Target 类
from torch.utils._sympy.interp import sympy_interp   # 从 torch.utils._sympy.interp 模块导入 sympy_interp 函数

log = logging.getLogger(__name__)   # 获取当前模块的日志记录器对象

try:
    import z3  # type: ignore[import]
    
    # Translation Validation for Dynamo guards
    # ========================================
    #
    # Checks whether optimizations applied to the collected guards are
    # valid. In other words, whether the guard function we actually run
    # does not have false positives (unsound).
    #
    # In order to do so, we build the guards using 2 different information
    # attached to each 'SymNode':
    #   1. SymPy expressions
    #   2. FX nodes
    #
    # SymPy expressions have implicit optimizations baked within itself,
    # which may have a few bugs. On the other hand, we build the FX graph
    # manually, with no optimizations enabled. This gives us access to
    # the "ground truth".
    #
    # We then convert into Z3 expressions both the SymPy expressions
    # (see [Note: SympyToZ3]) that reach 'ShapeEnv.produce_guards' function
    # and the FX nodes (see [Note: PopulateValidator]) that go through
    # 'ShapeEnv.evaluate_expr' function. Finally, we run the validation.
    # (see [Note: TranslationValidator])

    # Better Z3 to string implementation (for a small fraction of Z3).
    #
    # Here are the things we clean before showing the Z3 expression:
    #   - Rename a few ops (e.g. "Distinct" ==> "!=")
    #
    #   - Ignore ToInt and ToReal operations:
    #     usually they don't really matter
    #
    #   - Transform (ToInt (/ ...)) into (idiv ...):
    #     this is the pattern for floor division
    #
    #   - Collect a chain of the same operations into one

except ImportError:
    pass   # 如果导入 z3 失败，则什么也不做
    def z3str(e: z3.ExprRef) -> str:
        # 确保传入的表达式是有效的 Z3 表达式
        assert z3.is_expr(e), f"unsupported expression type: {e}"

        def get_args_str(e: z3.ExprRef) -> List[str]:
            # 获取表达式 e 的所有参数的字符串表示形式列表
            return [z3str(e.arg(i)) for i in range(e.num_args())]

        # 首先，简化给定的表达式。
        # 使用重写规则进行简化，所以不应该花费太长时间。
        e = z3.simplify(e)

        # 只支持函数应用
        # 即使 Z3 中的“变量”实际上也是函数应用
        if not z3.is_app(e):
            raise ValueError(f"can't print Z3 expression: {e}")

        # 如果是整数值或有理数值，直接返回其字符串表示
        if z3.is_int_value(e) or z3.is_rational_value(e):
            return e.as_string()  # type: ignore[attr-defined]

        # 获取表达式的声明和类型
        decl = e.decl()
        kind = decl.kind()
        op = str(decl)
        args = get_args_str(e)

        # 根据操作类型进行不同的处理
        if kind == z3.Z3_OP_POWER:
            op = "pow"

        elif kind in (z3.Z3_OP_ADD, z3.Z3_OP_MUL):
            # 收集 ADD 和 MUL 链的参数
            # 因为它们是可结合的，所以是安全的

            def collect_str_args(e):
                if not (z3.is_app(e) and e.decl().kind() == kind):
                    return [z3str(e)]
                else:
                    return [
                        x
                        for i in range(e.num_args())
                        for x in collect_str_args(e.arg(i))
                    ]

            args = collect_str_args(e)

        elif kind == z3.Z3_OP_NOT:
            # 恢复 z3.simplify 应用的一些转换
            #   - a != b ==> (Not (== a b)) ==> (!= a b)
            #   - a < b ==> (Not (<= b a)) ==> (> b a)
            #   - a > b ==> (Not (<= a b)) ==> (> a b)

            assert e.num_args() == 1
            arg = e.arg(0)

            assert z3.is_app(arg)
            argkind = arg.decl().kind()

            logic_inverse = {
                z3.Z3_OP_EQ: "!=",
                z3.Z3_OP_LE: ">",
                z3.Z3_OP_GE: "<",
            }

            if argkind in logic_inverse:
                op = logic_inverse[argkind]
                args = get_args_str(arg)

        elif kind in (z3.Z3_OP_TO_INT, z3.Z3_OP_TO_REAL):
            assert e.num_args() == 1
            argstr = z3str(e.arg(0))

            # 检查是否是整数除法模式
            if argstr.startswith("(/"):
                return "(idiv" + argstr[2:]

            # 否则，只需忽略它
            return argstr

        elif kind == z3.Z3_OP_UNINTERPRETED:
            assert e.num_args() == 0
            return str(decl)

        # 构造最终的表达式字符串
        string = op + " " + " ".join(args)
        return f"({string.rstrip()})"

    # Python 语义作为 Z3 表达式的实现。
    #
    # Z3 Real-Int 理论的运算符与 Python 的语义有所不同。
    # 因此，为了正确地使用 Z3，我们需要实现我们在 Python 中依赖的语义。
    @dataclass
    # 将可调用对象提升为可在 Z3 中使用的形式。
    #
    # 这个函数用于替换给定的操作符 'op'，使其：
    #
    #   1. 将参数转换为 Z3（即将它们变成 Z3 的元素）
    #
    #   2. 调用与 'op' 对应的操作，但使用 Z3 元素进行操作（如果操作本身就能在 Z3 下工作，则保持不变）
    def z3op(op: Callable, validator: "TranslationValidator") -> Callable:
        # 拥有布尔值作为参数的操作符。
        # 这是必要的，因为某些 FX 节点的参数是字面整数，而不是布尔值。因此，当此标志设置时，
        # 我们也将整数转换为布尔值。
        boolean_ops = {operator.not_, operator.and_, operator.or_}
        as_bool = op in boolean_ops

        # 将函数提升到 'z3.ExprRef' 域中。
        def lift(func):
            def wrap(a) -> z3.ExprRef:
                if isinstance(a, (z3.ArithRef, z3.BoolRef)):
                    return a
                # 如果是支持的类型之一，将其转换为 Z3 值。
                if isinstance(a, bool) or (as_bool and isinstance(a, int)):
                    return z3.BoolVal(bool(a))
                if isinstance(a, (int, sympy.Integer)):
                    return z3.IntVal(int(a))
                if isinstance(a, (float, sympy.Float)):
                    return z3.RealVal(float(a))
                raise ValueError(f"can't lift type: {type(a)}")

            @functools.wraps(func)
            def wrapper(*args):
                # 将参数提升为 Z3 元素的列表。
                wrapped_args = (wrap(a) for a in args)
                # 在 Z3 表达式上运行函数。
                return func(*wrapped_args)

            return wrapper

        # 创建 Z3 操作对象。
        ops = _Z3Ops(validator)
        
        # 替换映射，将给定的操作符映射到对应的 Z3 函数或提升后的函数。
        replacement_map = {
            # 操作符模块。
            operator.not_: lift(z3.Not),
            operator.and_: lift(z3.And),
            operator.or_: lift(z3.Or),
            operator.floordiv: lift(ops.floordiv),
            operator.truediv: lift(ops.div),
            operator.mod: lift(ops.mod),
            operator.abs: lift(ops.abs),
            builtins.round: lift(ops.round_to_int),

            # 数学模块。
            math.ceil: lift(ops.ceil),
            math.floor: lift(ops.floor),

            # Torch 模块。
            torch.sym_float: lift(ops.to_real),
            torch.sym_max: lift(ops.max),
            torch.sym_min: lift(ops.min),
            torch.sym_ite: lift(lambda b, t, f: t if b else f),
            torch._sym_sqrt: lift(ops.sqrt),  # 类型: 忽略[属性定义]
            # 不提升，因为我们只将此函数用作标记，用于将表达式添加为验证器的输入。
            torch._assert: torch._assert,
        }

        # 如果操作符 'op' 在替换映射中，则返回对应的替换函数，否则返回提升后的函数。
        return replacement_map[op] if op in replacement_map else lift(op)

    # 处理一个 FX 图，填充给定的验证器。
    #
    # [注意：PopulateValidator]
    # 这个类遍历 FX 图中的每个节点，将它们转换为 Z3 世界中的表达式。
    #
    # 每当它找到一个 'torch._assert' 的 call_function 操作时，
    # 将对应参数的 Z3 表达式作为验证器的输入添加进去。
    class PopulateValidator(torch.fx.Interpreter):
        
        # 初始化方法，接受一个 torch.fx.Graph 对象和一个 TranslationValidator 对象作为参数。
        def __init__(self, graph: torch.fx.Graph, validator: "TranslationValidator"):
            # 引用翻译验证器。
            self.validator = validator
            
            # 构建图模块并调用 `Interpreter` 构造函数。
            module = torch.fx.GraphModule(root={}, graph=graph)
            super().__init__(module, garbage_collect_values=True)

        # 占位符方法，接受一个 Target 对象、一组参数 args 和一个 kwargs 字典作为参数，并返回任意类型。
        def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
            # 获取当前 meta 数据中的符号信息。
            symbol = fx_traceback.get_current_meta()["symbol"]
            # 返回符号对应的 Z3 变量。
            return self.validator.z3var(symbol)

        # 调用函数方法，接受一个 Target 对象、一组参数 args 和一个 kwargs 字典作为参数，并返回任意类型。
        def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
            # 如果目标不是 torch._assert，则执行父类的 call_function 方法来执行节点目标函数。
            if target != torch._assert:
                return super().call_function(z3op(target, self.validator), args, kwargs)  # type: ignore[arg-type]
            
            # 如果是 torch._assert，则将第一个参数对应的 Z3 表达式作为验证器输入添加进去。
            assert len(args) == 1, f"expected 1 argument on assertion. Got: {len(args)} "
            self.validator.add_source_expr(args[0])  # type: ignore[arg-type]

    # 将 SymPy 表达式翻译为 Z3 表达式。
    #
    # [注意: SympyToZ3]
    # 在翻译时，所有出现在 SymPy 表达式中的自由变量必须已经映射到 Z3 整数变量。
    # Dynamo 保护翻译验证器。
    #
    # [注意: TranslationValidator]
    # 验证由 'ShapeEnv.produce_guards' 生成的保护是否有效。
    # 即: 是否只有在原始、未优化的（source）保护为 TRUE 时，目标（target）保护才为 TRUE。
    #
    # 更具体地说，给定 'source' 和 'target' 保护表达式，我们希望检查以下表达式是否成立:
    #
    # Not(And(source)) AND And(target)
    #
    # 即是否存在自由变量的赋值使得相反发生：target 为 TRUE，但是 source 为 FALSE。
except ImportError:
    # 如果导入失败，则表示 Z3 未安装
    _HAS_Z3 = False

    # 定义在此模块中导出的所有符号
    __all__ = [
        "translation_validation_enabled", "translation_validation_timeout",
        "ValidationException", "BisectValidationException",
    ]

else:
    # 如果成功导入，则表示 Z3 已安装
    _HAS_Z3 = True

    # 定义在此模块中导出的所有符号，包括 Z3 相关的内容
    __all__ = [
        "z3str", "z3op", "PopulateValidator", "SympyToZ3", "TranslationValidator",
        "translation_validation_enabled", "translation_validation_timeout",
        "ValidationException", "BisectValidationException",
    ]

# 导入 torch.fx.experimental 模块中的 _config 模块
from torch.fx.experimental import _config as config


def translation_validation_enabled() -> bool:
    # 检查每次调用此函数时，如果启用了翻译验证（translation validation），则确保 Z3 已安装
    _assert_z3_installed_if_tv_set()
    return _HAS_Z3 and config.translation_validation


def translation_validation_timeout() -> int:
    # 返回翻译验证超时设置
    return config.translation_validation_timeout


def _assert_z3_installed_if_tv_set():
    # 如果启用了翻译验证但未安装 Z3，则抛出异常
    assert _HAS_Z3 or not config.translation_validation, (
        "translation validation requires Z3 package. Please, either install "
        "z3-solver or disable translation validation."
    )


class ValidationException(TorchDynamoException):
    def __init__(self, model, assertions, target_exprs, failed_source_exprs):
        # 在初始化时确保 Z3 已安装
        assert _HAS_Z3

        # 定义辅助函数，用于生成字符串表示
        def symbolstr(sym) -> str:
            return f"{sym}: {model[sym]}"

        def joinlines(xs) -> str:
            return "\n".join(f"  ==> {x}" for x in xs)

        # 生成模型、断言、目标表达式和失败源表达式的字符串表示
        model_str = joinlines(sorted(map(symbolstr, model)))
        assertions_str = joinlines(sorted(map(z3str, assertions)))
        target_exprs_str = joinlines(sorted(map(z3str, target_exprs)))
        failed_source_exprs_str = joinlines(sorted(map(z3str, failed_source_exprs)))

        # 设置异常消息和详细信息
        self.msg = "translation validation failed."
        self.details = f"""\
Model:
{model_str}

Assertions:
{assertions_str}

Target Expressions:
{target_exprs_str}

Failed Source Expressions:
{failed_source_exprs_str}"""

    def __str__(self):
        # 返回异常的字符串表示
        return f"{self.msg}\n\n{self.details}"


class BisectValidationException(TorchDynamoException):
    def __init__(self, validation_exc, expr, failed_action, traced_node):
        # 设置异常消息和详细信息
        self.msg = f"translation validation failed when {failed_action}: {expr}"
        self.details = f"""\
Failure occurred while running node:
    {traced_node.format_node()}

{validation_exc.details}"""

    def __str__(self):
        # 返回异常的字符串表示
        return f"{self.msg}\n\n{self.details}"

# 在模块加载时检查 Z3 是否安装，以便启用翻译验证
_assert_z3_installed_if_tv_set()

# 翻译验证的二分法
#
# 将在 shape_env FX 图中记录的 torch._assert 节点进行二分，抛出最早的 ValidationException。
#
# 由于 ShapeEnv.evaluate_expr 调用添加了保护，可能会发生一些静默的简化错误。
# 此函数试图确切地找出从验证角度看出了哪些点出了问题。
def bisect(shape_env):
    # 导入需要的模块
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SHAPEENV_EVENT_KEY, CURRENT_NODE_KEY
    from torch.fx.experimental.recording import FakeTensorMeta, ShapeEnvEvent, replay_shape_env_events

    events = shape_env.events  # 从 shape_env 中获取事件列表

    # 根据节点获取其关联的 ShapeEnvEvent 对象
    def get_node_event(node: torch.fx.Node) -> ShapeEnvEvent:
        assert SHAPEENV_EVENT_KEY in node.meta  # 断言确保节点的元数据中包含 SHAPEENV_EVENT_KEY
        return events[node.meta[SHAPEENV_EVENT_KEY]]  # 返回与节点关联的 ShapeEnvEvent 对象

    # 使用给定的 shape_env 参数创建一个新的 fake 对象，更新所有符号值的 ShapeEnv 引用
    #
    # 这是为了避免使用“来自未来”的 ShapeEnv 简化符号表达式，可能会有不同的替换集合。
    def new_with_shape_env(shape_env: ShapeEnv, fake) -> Any:
        if isinstance(fake, int):
            return fake
        if isinstance(fake, torch.SymInt):
            return torch.SymInt(fake.node.with_shape_env(shape_env))
        assert isinstance(fake, FakeTensorMeta)
        return FakeTensorMeta(
            tuple(new_with_shape_env(shape_env, s) for s in fake.size()),
            tuple(new_with_shape_env(shape_env, s) for s in fake.stride()),
            new_with_shape_env(shape_env, fake.storage_offset()),
            fake.is_nested,
        )

    # 检查给定的 shape_env 在调用 produce_guards 时是否失败
    def check_shapeenv_fails(shape_env: ShapeEnv, tracked_fakes: Optional[List[Any]]) -> Optional[ValidationException]:
        assert tracked_fakes is not None  # 断言确保 tracked_fakes 不为空
        try:
            # 这个 produce_guards 调用是尽力复制，因为我们不填充 EqualityConstraint 列表的原因：
            # 我们还需要保存 OutputGraph.tracked_fakes_id_to_source。
            shape_env.produce_guards(
                [new_with_shape_env(shape_env, a.fake) for a in tracked_fakes],  # 使用新的 shape_env 更新 tracked_fakes 中的 fake 对象
                [a.source for a in tracked_fakes],  # 获取 tracked_fakes 中每个元素的源信息
                input_contexts=[a.symbolic_context for a in tracked_fakes],  # 获取 tracked_fakes 中每个元素的符号上下文
            )
            return None
        except ValidationException as e:
            return e  # 返回验证异常对象

    # 检查通过重播事件直到创建节点 node 时重建的 ShapeEnv 在调用 produce_guards 时是否失败
    def check_node_fails(node: torch.fx.Node) -> Optional[ValidationException]:
        number = node.meta[SHAPEENV_EVENT_KEY]  # 获取节点的 SHAPEENV_EVENT_KEY
        # 重建 shape_env 直到事件编号为 number 的位置
        shape_env = replay_shape_env_events(events[:number + 1])
        shape_env.graph.lint()  # 对 shape_env 的图进行检查
        return check_shapeenv_fails(shape_env, events[number].tracked_fakes)  # 检查重建后的 shape_env 是否失败

    last_exception = check_shapeenv_fails(shape_env, shape_env._snapshot_tracked_fakes())  # 检查当前 shape_env 是否失败

    if not last_exception:
        # 如果没有因 produce_guards 调用而失败
        # 停止并且不进行二分搜索
        log.info("translation validation succeeded: no errors found.")  # 记录信息：翻译验证成功，未发现错误
        return
    # 如果不需要记录事件或者配置禁用了二分法验证，则抛出最后一个记录的异常
    if not shape_env.should_record_events or config.translation_validation_no_bisect:
        raise last_exception

    # 存储每个二分点可能引发的异常
    exception = {}

    # 获取所有断言节点，用于动态形状的FX图的二分法验证
    assert_nodes = [node for node in shape_env.graph.nodes if node.target == torch._assert]

    # 准备二分搜索的索引
    left, mid, right = 0, 0, len(assert_nodes) - 1

    while left < right:
        mid = (left + right) // 2

        node = assert_nodes[mid]
        log.debug("bisecting at %s: %s", mid, get_node_event(node))

        # 检查新的 shape_env 是否引发 ValidationException 异常
        exception[mid] = check_node_fails(node)

        if exception[mid]:
            right = mid
        else:
            left = mid + 1

    # 确保 left 索引在异常字典中，且对应的异常是 ValidationException 类型
    assert left in exception and isinstance(exception[left], ValidationException)

    # 获取引发异常的节点及其事件
    node = assert_nodes[left]
    event = get_node_event(node)

    # 根据事件类型确定失败的操作
    if event.is_evaluate_expr():
        failed_action = "evaluating"
    else:
        assert event.is_defer_runtime_assert(), f"unexpected event type: {event}"
        failed_action = "adding runtime assert"

    # 确保事件参数不为 None，并且至少有两个位置参数
    args = event.args
    assert args is not None
    assert len(args) >= 2, (
        f"bisecting expects {event.name} to have at least 2 positional arguments. "
        f"Got: {len(args)}"
    )
    # 确保第二个参数是 SymPy 表达式
    assert isinstance(args[1], sympy.Basic), (
        f"bisecting expects {event.name} to have a SymPy expression as its second argument. "
        f"Got: {type(args[1])}"
    )

    # 抛出 BisectValidationException 异常，包含相关信息
    raise BisectValidationException(
        exception[left],
        expr=args[1],
        failed_action=failed_action,
        traced_node=node.meta[CURRENT_NODE_KEY],
    )
```