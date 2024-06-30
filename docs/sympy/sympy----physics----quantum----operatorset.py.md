# `D:\src\scipysrc\sympy\sympy\physics\quantum\operatorset.py`

```
# 这个模块用于将运算符映射到它们对应的本征态以及反之

from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
                                             PositionKet3D)
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
                                        JzKet)

__all__ = [
    'operators_to_state',
    'state_to_operators'
]

# state_mapping 存储了状态与它们关联的运算符或运算符元组之间的映射关系
# 当新的状态-运算符对被创建时，这个字典应该被更新
# 条目的形式为 PxKet : PxOp 或者类似 3DKet : (ROp, ThetaOp, PhiOp)
state_mapping = { JxKet: frozenset((J2Op, JxOp)),
                  JyKet: frozenset((J2Op, JyOp)),
                  JzKet: frozenset((J2Op, JzOp)),
                  Ket: Operator,
                  PositionKet3D: frozenset((XOp, YOp, ZOp)),
                  PxKet: PxOp,
                  XKet: XOp }

# op_mapping 用于反向映射，将运算符映射回状态
op_mapping = {v: k for k, v in state_mapping.items()}


def operators_to_state(operators, **options):
    """ 返回给定运算符或运算符集的本征态

    这是一个全局函数，用于将运算符类映射到它们关联的状态。它接受一个
    运算符或一组运算符，并返回与它们关联的状态。

    这个函数可以处理给定运算符的实例或类本身（即 XOp() 和 XOp）。

    有多种使用情况需要考虑：

    1) 传递了一个类或一组类：首先，我们尝试为这些运算符实例化默认实例。
    如果失败，则简单地返回类本身。如果成功实例化了默认实例，则尝试在
    运算符实例上调用 state._operators_to_state。如果失败，则返回类本身。
    否则，返回由 _operators_to_state 返回的实例。

    2) 传递了一个实例或一组实例：在这种情况下，调用 state._operators_to_state。
    如果失败，则返回一个状态类。如果方法返回一个实例，则返回该实例。

    在这两种情况下，如果运算符类或集合在 state_mapping 字典中不存在，则返回 None。

    参数
    ==========

    """
    # 检查 operators 是否为 Operator 类型或者 Operator 的子类的集合
    if not (isinstance(operators, (Operator, set)) or issubclass(operators, Operator)):
        # 如果不是 Operator 或其子类的集合，则抛出未实现的错误
        raise NotImplementedError("Argument is not an Operator or a set!")

    # 如果 operators 是一个集合
    if isinstance(operators, set):
        # 遍历集合中的每个元素 s
        for s in operators:
            # 检查每个元素是否为 Operator 类型或 Operator 的子类
            if not (isinstance(s, Operator)
                   or issubclass(s, Operator)):
                # 如果不是，则抛出未实现的错误
                raise NotImplementedError("Set is not all Operators!")

        # 将 operators 转换为不可变集合 ops
        ops = frozenset(operators)

        # 如果 ops 在 op_mapping 中存在
        if ops in op_mapping:
            # 尝试使用 ops 中默认实例的对象，如果失败则返回类本身
            try:
                # 创建 ops 中每个类的默认实例列表
                op_instances = [op() for op in ops]
                # 调用 _get_state 函数，传入 op_mapping[ops]、op_instances 和 options
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                # 如果创建实例失败，则返回 op_mapping[ops]
                ret = op_mapping[ops]

            # 返回结果 ret
            return ret
        else:
            # 如果 ops 不在 op_mapping 中，则创建 ops 类型的列表 tmp
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)

            # 如果 classes 在 op_mapping 中存在
            if classes in op_mapping:
                # 调用 _get_state 函数，传入 op_mapping[classes]、ops 和 options
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                # 如果 classes 不在 op_mapping 中，则返回 None
                ret = None

            # 返回结果 ret
            return ret
    else:
        # 如果 operators 不是集合类型

        # 如果 operators 在 op_mapping 中存在
        if operators in op_mapping:
            try:
                # 创建 operators 的默认实例 op_instance
                op_instance = operators()
                # 调用 _get_state 函数，传入 op_mapping[operators]、op_instance 和 options
                ret = _get_state(op_mapping[operators], op_instance, **options)
            except NotImplementedError:
                # 如果创建实例失败，则返回 op_mapping[operators]
                ret = op_mapping[operators]

            # 返回结果 ret
            return ret
        elif type(operators) in op_mapping:
            # 如果 operators 的类型在 op_mapping 中存在，则调用 _get_state 函数
            return _get_state(op_mapping[type(operators)], operators, **options)
        else:
            # 如果 operators 不在 op_mapping 中，则返回 None
            return None
def state_to_operators(state, **options):
    """ Returns the operator or set of operators corresponding to the
    given eigenstate

    A global function for mapping state classes to their associated
    operators or sets of operators. It takes either a state class
    or instance.

    This function can handle both instances of a given state or just
    the class itself (i.e. both XKet() and XKet)

    There are multiple use cases to consider:

    1) A state class is passed: In this case, we first try
    instantiating a default instance of the class. If this succeeds,
    then we try to call state._state_to_operators on that instance.
    If the creation of the default instance or if the calling of
    _state_to_operators fails, then either an operator class or set of
    operator classes is returned. Otherwise, the appropriate
    operator instances are returned.

    2) A state instance is returned: Here, state._state_to_operators
    is called for the instance. If this fails, then a class or set of
    operator classes is returned. Otherwise, the instances are returned.

    In either case, if the state's class does not exist in
    state_mapping, None is returned.

    Parameters
    ==========

    state: StateBase class or instance (or subclasses)
         The class or instance of the state to be mapped to an
         operator or set of operators

    **options:
         Additional options to be passed to _get_ops function

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra
    >>> from sympy.physics.quantum.operatorset import state_to_operators
    >>> from sympy.physics.quantum.state import Ket, Bra
    >>> state_to_operators(XKet)
    X
    >>> state_to_operators(XKet())
    X
    >>> state_to_operators(PxKet)
    Px
    >>> state_to_operators(PxKet())
    Px
    >>> state_to_operators(PxBra)
    Px
    >>> state_to_operators(XBra)
    X
    >>> state_to_operators(Ket)
    O
    >>> state_to_operators(Bra)
    O
    """

    # Check if the input is either a StateBase instance or a subclass
    if not (isinstance(state, StateBase) or issubclass(state, StateBase)):
        raise NotImplementedError("Argument is not a state!")

    # Check if the state class is already in the state_mapping dictionary
    if state in state_mapping:  # state is a class
        # Attempt to create a default instance of the state class
        state_inst = _make_default(state)
        try:
            # Try to get the operators for the instance using _get_ops function
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state]), **options)
        except (NotImplementedError, TypeError):
            # If instantiation or _get_ops fails, return the mapped operator class or set
            ret = state_mapping[state]
    # Check if the type of state is in state_mapping (for instances of state)
    elif type(state) in state_mapping:
        # Get the operators for the instance using _get_ops function
        ret = _get_ops(state,
                       _make_set(state_mapping[type(state)]), **options)
    # Special case for instances of BraBase where their dual class is in state_mapping
    elif isinstance(state, BraBase) and state.dual_class() in state_mapping:
        # Get the operators for the instance using _get_ops function
        ret = _get_ops(state,
                       _make_set(state_mapping[state.dual_class()]))
    # 如果 state 是 BraBase 类的子类，并且其对偶类存在于 state_mapping 中
    elif issubclass(state, BraBase) and state.dual_class() in state_mapping:
        # 创建一个默认的 state_inst 实例
        state_inst = _make_default(state)
        try:
            # 尝试获取 state_inst 的操作，使用其对偶类作为 state_mapping 的键
            ret = _get_ops(state_inst, _make_set(state_mapping[state.dual_class()]))
        except (NotImplementedError, TypeError):
            # 如果获取操作时出现 NotImplementedError 或 TypeError，则直接使用 state_mapping[state.dual_class()] 的值
            ret = state_mapping[state.dual_class()]
    else:
        # 如果 state 不满足上述条件，则将 ret 设为 None
        ret = None

    # 返回结果，将 ret 转换为集合并返回
    return _make_set(ret)
# 尝试执行表达式 `expr`，如果抛出 TypeError 异常，则将 `expr` 作为返回值
def _make_default(expr):
    try:
        ret = expr()
    except TypeError:
        ret = expr

    return ret


# 尝试从状态类 `state_class` 的操作器中获取状态实例 `state_class._operators_to_state(ops, **options)`
# 如果抛出 NotImplementedError 异常，则返回 `state_class` 的默认实例 `_make_default(state_class)`
def _get_state(state_class, ops, **options):
    try:
        ret = state_class._operators_to_state(ops, **options)
    except NotImplementedError:
        ret = _make_default(state_class)

    return ret


# 尝试从状态实例 `state_inst` 中获取操作器实例 `state_inst._state_to_operators(op_classes, **options)`
# 如果抛出 NotImplementedError 异常，则根据 `op_classes` 的类型返回默认值 `_make_default`
# 如果 `op_classes` 是集合类型（set, tuple, frozenset），则返回其中每个元素的默认值构成的元组
# 否则返回 `op_classes` 的默认值
def _get_ops(state_inst, op_classes, **options):
    try:
        ret = state_inst._state_to_operators(op_classes, **options)
    except NotImplementedError:
        if isinstance(op_classes, (set, tuple, frozenset)):
            ret = tuple(_make_default(x) for x in op_classes)
        else:
            ret = _make_default(op_classes)

    # 如果 `ret` 是集合类型且长度为1，则返回集合中的唯一元素
    if isinstance(ret, set) and len(ret) == 1:
        return ret[0]

    return ret


# 根据 `ops` 的类型构造一个集合对象
# 如果 `ops` 是元组、列表或不可变集合（frozenset），则返回一个包含 `ops` 所有元素的集合
# 否则直接返回 `ops`
def _make_set(ops):
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops
```