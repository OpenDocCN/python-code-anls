# `.\pytorch\torch\_dynamo\bytecode_transformation.py`

```
# 设置允许未标记的定义，用于类型检查
# 导入必要的模块和库
import copy  # 导入 copy 模块，用于复制对象
import dataclasses  # 导入 dataclasses 模块，用于创建数据类
import dis  # 导入 dis 模块，用于反汇编 Python 字节码
import itertools  # 导入 itertools 模块，用于创建迭代器
import sys  # 导入 sys 模块，提供与 Python 解释器相关的功能
import types  # 导入 types 模块，用于操作 Python 类型系统
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple, Union  # 导入 typing 模块，提供类型提示支持

from .bytecode_analysis import (
    get_indexof,  # 导入自定义模块中的函数 get_indexof
    propagate_line_nums,  # 导入自定义模块中的函数 propagate_line_nums
    remove_extra_line_nums,  # 导入自定义模块中的函数 remove_extra_line_nums
    stacksize_analysis,  # 导入自定义模块中的函数 stacksize_analysis
)


@dataclasses.dataclass
class InstructionExnTabEntry:
    """异常表条目，记录了异常处理的指令范围和相关信息"""

    start: "Instruction"  # 异常处理起始指令
    end: "Instruction"  # 异常处理结束指令
    target: "Instruction"  # 异常处理目标指令
    depth: int  # 异常处理深度
    lasti: bool  # 是否为最后一条指令

    def __repr__(self) -> str:
        """返回异常表条目的字符串表示形式"""
        return (
            f"InstructionExnTabEntry(start={self.start.short_inst_repr()}, "
            f"end={self.end.short_inst_repr()}, "
            f"target={self.target.short_inst_repr()}, "
            f"depth={self.depth}, lasti={self.lasti})"
        )

    def __eq__(self, o) -> bool:
        """比较两个异常表条目是否相等"""
        return (
            self.start is o.start
            and self.end is o.end
            and self.target is o.target
            and self.depth == o.depth
            and self.lasti == o.lasti
        )


@dataclasses.dataclass
class Instruction:
    """可变版本的 dis.Instruction"""

    opcode: int  # 操作码
    opname: str  # 操作名称
    arg: Optional[int]  # 参数值（可选）
    argval: Any  # 参数值的具体内容
    offset: Optional[int] = None  # 偏移量（可选）
    starts_line: Optional[int] = None  # 起始行号（可选）
    is_jump_target: bool = False  # 是否为跳转目标
    positions: Optional["dis.Positions"] = None  # 位置信息（可选）
    target: Optional["Instruction"] = None  # 目标指令（可选）
    exn_tab_entry: Optional[InstructionExnTabEntry] = None  # 异常表条目（可选）

    def __hash__(self) -> int:
        """返回指令对象的哈希值"""
        return id(self)

    def __eq__(self, other) -> bool:
        """比较两个指令对象是否相等"""
        return id(self) == id(other)

    def short_inst_repr(self) -> str:
        """返回指令对象的简短表示形式"""
        return f"Instruction(opname={self.opname}, offset={self.offset})"


def convert_instruction(i: dis.Instruction) -> Instruction:
    """将 dis.Instruction 转换为可变版本的 Instruction 对象"""
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
        getattr(i, "positions", None),
    )


class _NotProvided:
    """未提供特定值的占位符类"""

    def __repr__(self) -> str:
        """返回表示未提供的字符串"""
        return "_NotProvided"


def inst_has_op_bits(name):
    """检查指令名称是否具有操作位"""
    return (sys.version_info >= (3, 11) and name == "LOAD_GLOBAL") or (
        sys.version_info >= (3, 12) and name in ("LOAD_ATTR", "LOAD_SUPER_ATTR")
    )


def create_instruction(
    name, *, arg=None, argval=_NotProvided, target=None
) -> Instruction:
    """
    创建指令对象，根据指定的名称和参数。

    如果 `arg` 未提供，将从 `argval` 或 `target` 计算出 `arg`。

    对于指令 LOAD_GLOBAL、LOAD_ATTR（3.12+）、LOAD_SUPER_ATTR，参数位会修改指令的行为，
    因此允许设置 `arg` 和 `argval`。此处的 `arg` 预期是 co_consts 中的值。

    """
    # 创建并返回指令对象
    return Instruction(
        name,
        opcode=name,
        arg=arg,
        argval=argval,
        target=target
    )
    the op bits and the true value of `arg` will be computed during assembly.
    If `arg` is not set, the bits are assumed to be 0.
    """

    # 如果指令具有操作位（op bits），允许同时指定 arg 和 argval
    if inst_has_op_bits(name):
        # 如果指定了 target，抛出运行时错误
        if target is not None:
            raise RuntimeError("target cannot be specified for instruction")
        # 如果 arg 未设置，则默认为 0
        if arg is None:
            arg = 0
    else:
        # 计算非空参数的数量
        cnt = (arg is not None) + (argval is not _NotProvided) + (target is not None)
        # 如果超过一个参数不为空，则抛出运行时错误
        if cnt > 1:
            raise RuntimeError(
                "only one of arg, argval, and target can be not None/_NotProvided"
            )
    # 如果 arg 不为空且不是整数，则抛出运行时错误
    if arg is not None and not isinstance(arg, int):
        raise RuntimeError("instruction arg must be int or None")
    # 返回一个 Instruction 对象，包括操作码、操作名称、arg、argval 和 target
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )
# Python 3.11 remaps
# 根据 Python 版本选择不同的指令名称创建跳转指令
def create_jump_absolute(target) -> Instruction:
    inst = "JUMP_FORWARD" if sys.version_info >= (3, 11) else "JUMP_ABSOLUTE"
    return create_instruction(inst, target=target)


# 根据 Python 版本选择不同的指令名称创建复制栈顶元素指令
def create_dup_top() -> Instruction:
    if sys.version_info >= (3, 11):
        return create_instruction("COPY", arg=1)
    return create_instruction("DUP_TOP")


# 根据 Python 版本生成不同的指令序列，用于将栈顶元素旋转到第 n 个位置
def create_rot_n(n) -> List[Instruction]:
    """
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    """
    if n <= 1:
        # 不进行旋转操作
        return []

    if sys.version_info >= (3, 11):
        # 旋转可以用一系列交换操作表示
        # 例如，旋转 3 相当于交换 3, 2
        return [create_instruction("SWAP", arg=i) for i in range(n, 1, -1)]

    # 确保所需的旋转功能存在
    if sys.version_info < (3, 8) and n >= 4:
        raise AttributeError(f"rotate {n} not supported for Python < 3.8")
    if sys.version_info < (3, 10) and n >= 5:
        raise AttributeError(f"rotate {n} not supported for Python < 3.10")

    if n <= 4:
        return [create_instruction("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])]
    return [create_instruction("ROT_N", arg=n)]


# 根据 Python 版本在指令序列的前后添加 PUSH_NULL 指令
def add_push_null(
    inst_or_insts: Union[Instruction, List[Instruction]],
) -> List[Instruction]:
    """
    Appends or prepends a PUSH_NULL instruction to `inst_or_insts`,
    depending on Python version. Used when you know that
    `inst_or_insts` generates a callable that will be called.

    NOTE: Assumes `inst_or_insts` is a single instruction or sequence of
    instructions that pushes exactly 1 object to the stack that is to
    be called. It is important that you include ALL instructions that
    construct the callable - not just the first instruction/a prefix.

    Will attempt to use the NULL push bit for instructions
    with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
    In this case, instructions WILL be modified.
    """
    if isinstance(inst_or_insts, Instruction):
        insts = [inst_or_insts]
    else:
        insts = inst_or_insts

    # 检查指令是否设置了指定位
    def inst_has_bit_set(idx):
        assert insts[idx].arg is not None
        return insts[idx].arg & 1 == 1

    # 设置指令的指定位
    def set_inst_bit(idx):
        assert insts[idx].arg is not None
        insts[idx].arg |= 1
    # 如果 Python 版本大于等于 3.13
    if sys.version_info >= (3, 13):
        # 如果最后一条指令有操作位(inst_has_op_bits(insts[-1].opname))，并且最后一条指令的位未设置(inst_has_bit_set(-1)为假)
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            # 设置最后一条指令的位
            set_inst_bit(-1)
        else:
            # 在指令列表末尾添加一个 PUSH_NULL 指令
            insts = insts + [create_instruction("PUSH_NULL")]
    # 如果 Python 版本大于等于 3.12 但小于 3.13
    elif sys.version_info >= (3, 12):
        # 如果最后一条指令有操作位，并且最后一条指令的位未设置
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            # 设置最后一条指令的位
            set_inst_bit(-1)
        # 如果第一条指令是 "LOAD_GLOBAL" 并且第一条指令的位未设置
        elif insts[0].opname == "LOAD_GLOBAL" and not inst_has_bit_set(0):
            # 设置第一条指令的位
            set_inst_bit(0)
        else:
            # 在指令列表开头添加一个 PUSH_NULL 指令
            insts = [create_instruction("PUSH_NULL")] + insts
    # 如果 Python 版本大于等于 3.11 但小于 3.12
    elif sys.version_info >= (3, 11):
        # 如果第一条指令有操作位，并且第一条指令的位未设置
        if inst_has_op_bits(insts[0].opname) and not inst_has_bit_set(0):
            # 设置第一条指令的位
            set_inst_bit(0)
        else:
            # 在指令列表开头添加一个 PUSH_NULL 指令
            insts = [create_instruction("PUSH_NULL")] + insts
    # 返回处理后的指令列表
    return insts
def add_push_null_call_function_ex(
    inst_or_insts: Union[Instruction, List[Instruction]],
) -> List[Instruction]:
    """
    Like add_push_null, but the low bit of LOAD_ATTR/LOAD_SUPER_ATTR
    is not set, due to an expected CALL_FUNCTION_EX instruction.
    """
    if isinstance(inst_or_insts, Instruction):
        insts = [inst_or_insts]  # 将单个指令对象转换为列表形式
    else:
        insts = inst_or_insts  # 直接使用传入的指令列表

    if sys.version_info < (3, 11):
        return insts  # 如果 Python 版本低于 3.11，直接返回指令列表

    idx = -1 if sys.version_info >= (3, 13) else 0
    if insts[idx].opname == "LOAD_GLOBAL":  # 检查最后或第一个指令是否为 LOAD_GLOBAL
        assert insts[idx].arg is not None  # 确保指令参数不为空
        if insts[idx].arg & 1 == 0:  # 检查参数值是否为偶数
            insts[idx].arg |= 1  # 将参数值最低位设置为 1，用于 CALL_FUNCTION_EX
            return insts  # 返回更新后的指令列表

    if sys.version_info >= (3, 13):
        insts = insts + [create_instruction("PUSH_NULL")]  # 在指令列表末尾添加 PUSH_NULL 指令
    else:
        insts = [create_instruction("PUSH_NULL")] + insts  # 在指令列表开头添加 PUSH_NULL 指令

    return insts  # 返回更新后的指令列表
    # 检查 Python 版本是否大于等于 3.11
    if sys.version_info >= (3, 11):
        # 创建一个空列表用于存储指令
        output = []
        # 如果 push_null 为真，向 output 中添加 PUSH_NULL 指令
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 在 Python 版本大于等于 3.13 时，调整 rots 的值
            rots = nargs + 1 if sys.version_info >= (3, 13) else nargs + 2
            # 将 create_rot_n(rots) 返回的指令列表扩展到 output 中
            output.extend(create_rot_n(rots))
        # 在 Python 版本小于 3.12 时，向 output 中添加 PRECALL 指令
        if sys.version_info < (3, 12):
            output.append(create_instruction("PRECALL", arg=nargs))
        # 向 output 中添加 CALL 指令，arg 设置为 nargs
        output.append(create_instruction("CALL", arg=nargs))
        # 返回存储指令的 output 列表
        return output
    # 如果 Python 版本小于 3.11，则返回一个包含 CALL_FUNCTION 指令的列表
    return [create_instruction("CALL_FUNCTION", arg=nargs)]
# 创建调用方法指令的函数，返回一个指令对象列表
def create_call_method(nargs) -> List[Instruction]:
    # 如果 Python 版本大于等于 3.12，创建一个 CALL 指令
    if sys.version_info >= (3, 12):
        return [create_instruction("CALL", arg=nargs)]
    # 如果 Python 版本大于等于 3.11，创建 PRECALL 和 CALL 两个指令
    if sys.version_info >= (3, 11):
        return [
            create_instruction("PRECALL", arg=nargs),
            create_instruction("CALL", arg=nargs),
        ]
    # 否则创建一个 CALL_METHOD 指令
    return [create_instruction("CALL_METHOD", arg=nargs)]


# 创建加载方法指令的函数，返回一个指令对象
def create_load_method(name) -> Instruction:
    # 如果 Python 版本大于等于 3.12，创建一个 LOAD_ATTR 指令，设置 arg 参数为 1，argval 参数为 name
    if sys.version_info >= (3, 12):
        return create_instruction("LOAD_ATTR", arg=1, argval=name)
    # 否则创建一个 LOAD_METHOD 指令，设置 argval 参数为 name
    return create_instruction("LOAD_METHOD", argval=name)


# 创建设置 with 上下文的指令函数，返回一个指令对象
def create_setup_with(target) -> Instruction:
    # 如果 Python 版本大于等于 3.11，设置 opname 为 "BEFORE_WITH"，否则为 "SETUP_WITH"
    opname = "BEFORE_WITH" if sys.version_info >= (3, 11) else "SETUP_WITH"
    return create_instruction(opname, target=target)


# 创建交换栈元素的指令函数，返回一个指令对象列表
def create_swap(n) -> List[Instruction]:
    # 如果 Python 版本大于等于 3.11，创建一个 SWAP 指令
    if sys.version_info >= (3, 11):
        return [create_instruction("SWAP", arg=n)]
    # 否则在 Python < 3.11，SWAP 是一个宏展开成多个指令
    if n == 1:
        return []
    """
    e.g. swap "a" and "b" in this stack:
    0 a 1 2 3 b
    0 a [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b] -1
    0 a [1 2 3 b] b
    0 b a [1 2 3 b]
    0 b a [1 2 3 b] [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b] -1
    0 b [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a] reverse
    0 b [a 3 2 1] None
    0 b [a 3 2 1]
    0 b 1 2 3 a
    """
    # 创建多个指令来实现栈元素的交换
    return [
        create_instruction("BUILD_LIST", arg=n - 1),
        create_instruction("DUP_TOP"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("BINARY_SUBSCR"),
        create_instruction("ROT_THREE"),
        create_instruction("DUP_TOP"),
        create_instruction("ROT_THREE"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("STORE_SUBSCR"),
        create_instruction("DUP_TOP"),
        create_load_method("reverse"),
        *create_call_method(0),
        create_instruction("POP_TOP"),
        create_instruction("UNPACK_SEQUENCE", arg=n - 1),
    ]


# 创建 lnotab 写入函数，返回一个元组，包含 lnotab 列表和一个更新函数
def lnotab_writer(
    lineno: int, byteno: int = 0
) -> Tuple[List[int], Callable[[int, int], None]]:
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
    # 断言当前 Python 版本小于 3.10
    assert sys.version_info < (3, 10)
    # 初始化 lnotab 列表为空
    lnotab: List[int] = []

    # 定义一个更新函数 update，用来更新行号和字节偏移
    def update(lineno_new, byteno_new):
        nonlocal byteno, lineno
        # 循环直到行号和字节偏移都更新到目标值
        while byteno_new != byteno or lineno_new != lineno:
            byte_offset = max(0, min(byteno_new - byteno, 255))
            line_offset = max(-128, min(lineno_new - lineno, 127))
            # 断言偏移量不为零
            assert byte_offset != 0 or line_offset != 0
            byteno += byte_offset
            lineno += line_offset
            # 将更新后的字节偏移和行偏移加入 lnotab 列表
            lnotab.extend((byte_offset, line_offset & 0xFF))

    # 返回 lnotab 列表和 update 函数
    return lnotab, update
# 创建一个函数 linetable_310_writer，用于生成 Python 3.10 版本的行号表
def linetable_310_writer(first_lineno):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table for Python 3.10
    """
    # 断言当前 Python 版本为 3.10.x
    assert sys.version_info >= (3, 10) and sys.version_info < (3, 11)
    # 初始化行号表 linetable
    linetable: List[int] = []
    # 初始行号设定为参数传入的 first_lineno
    lineno = first_lineno
    # 初始化行号偏移和字节偏移
    lineno_delta = 0
    byteno = 0

    # 内部函数 _update 用于更新行号表
    def _update(byteno_delta, lineno_delta):
        # 只要字节偏移或行号偏移不为零，持续更新行号表
        while byteno_delta != 0 or lineno_delta != 0:
            # 计算字节偏移和行号偏移的范围
            byte_offset = max(0, min(byteno_delta, 254))
            line_offset = max(-127, min(lineno_delta, 127))
            # 断言字节偏移和行号偏移至少有一个不为零
            assert byte_offset != 0 or line_offset != 0
            # 减去已处理的字节偏移和行号偏移
            byteno_delta -= byte_offset
            lineno_delta -= line_offset
            # 将更新的字节偏移和行号偏移添加到行号表中
            linetable.extend((byte_offset, line_offset & 0xFF))

    # 函数 update 用于更新当前行号和字节偏移
    def update(lineno_new, byteno_new):
        nonlocal lineno, lineno_delta, byteno
        # 计算新的字节偏移和更新字节偏移
        byteno_delta = byteno_new - byteno
        byteno = byteno_new
        # 调用 _update 函数更新行号表
        _update(byteno_delta, lineno_delta)
        # 更新行号偏移和当前行号
        lineno_delta = lineno_new - lineno
        lineno = lineno_new

    # 函数 end 用于完成行号表的最终更新
    def end(total_bytes):
        _update(total_bytes - byteno, lineno_delta)

    # 返回行号表，更新函数 update 和结束函数 end
    return linetable, update, end


# 创建函数 encode_varint，用于将整数 n 编码为变长整数列表
def encode_varint(n: int) -> List[int]:
    """
    6-bit chunk encoding of an unsigned integer
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    """
    # 断言 n 为非负数
    assert n >= 0
    # 初始化变长整数列表 b
    b = [n & 63]
    n >>= 6
    # 循环编码 n 的每个 6 位块
    while n > 0:
        b[-1] |= 64
        b.append(n & 63)
        n >>= 6
    # 返回编码后的变长整数列表
    return b


# 创建函数 linetable_311_writer，用于生成 Python 3.11 版本的行号表
def linetable_311_writer(first_lineno: int):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    This is the internal format of the line number table for Python 3.11
    """
    # 断言当前 Python 版本为 3.11.x
    assert sys.version_info >= (3, 11)
    # 初始化行号表 linetable
    linetable = []
    # 初始行号设定为参数传入的 first_lineno
    lineno = first_lineno
    def update(positions: "dis.Positions", inst_size):
        nonlocal lineno  # 声明 lineno 变量为非局部变量，将在闭包中使用

        lineno_new = positions.lineno if positions else None  # 获取新的行号信息，如果 positions 存在则使用其行号，否则为 None

        def _update(delta, size):
            assert 0 < size <= 8  # 断言确保 size 大于 0 且不超过 8

            # 根据 positions 的存在和完整性，决定使用何种编码方式
            # 如果 positions 存在且其所有字段均不为 None，则使用长形式编码
            # 否则使用短形式编码
            other_varints: Tuple[int, ...] = ()
            if (
                positions
                and positions.lineno is not None
                and positions.end_lineno is not None
                and positions.col_offset is not None
                and positions.end_col_offset is not None
            ):
                linetable.append(0b1_1110_000 + size - 1)  # 添加长形式编码的表项
                # 列偏移需要加 1 的原因详见链接
                other_varints = (
                    positions.end_lineno - positions.lineno,
                    positions.col_offset + 1,
                    positions.end_col_offset + 1,
                )
            else:
                linetable.append(0b1_1101_000 + size - 1)  # 添加短形式编码的表项

            # 编码有符号整数 delta
            if delta < 0:
                delta = ((-delta) << 1) | 1
            else:
                delta <<= 1
            # 编码无符号整数 delta
            linetable.extend(encode_varint(delta))
            # 编码其他变长整数
            for n in other_varints:
                linetable.extend(encode_varint(n))

        if lineno_new is None:
            lineno_delta = 0
        else:
            lineno_delta = lineno_new - lineno  # 计算行号增量
            lineno = lineno_new  # 更新当前行号为新行号

        while inst_size > 8:
            _update(lineno_delta, 8)  # 更新表格内容，处理剩余的 8 字节指令大小
            inst_size -= 8  # 减去已处理的指令大小

        _update(lineno_delta, inst_size)  # 处理剩余不足 8 字节的指令大小

    return linetable, update  # 返回编码表格和更新函数
@dataclasses.dataclass
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


def encode_exception_table_varint(n: int) -> List[int]:
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
    assert n >= 0
    b = [n & 63]  # 取 n 的低 6 位作为初始值
    n >>= 6  # 将 n 右移 6 位
    while n > 0:
        b.append(n & 63)  # 继续取 n 的低 6 位，加入列表
        n >>= 6  # 再次将 n 右移 6 位
    b.reverse()  # 将列表反转，得到正确的 varint 编码顺序
    for i in range(len(b) - 1):
        b[i] |= 64  # 对于除最后一个元素外的每个元素，将第 7 位设为 1
    return b  # 返回 varint 编码的列表


def decode_exception_table_varint(bytes_iter: Iterator[int]) -> int:
    """
    Inverse of `encode_exception_table_varint`.
    """
    b = next(bytes_iter)  # 从迭代器中取下一个字节
    val = b & 63  # 取 b 的低 6 位作为初始值
    while b & 64:  # 如果 b 的第 7 位为 1，继续解析
        val <<= 6  # 左移 6 位
        b = next(bytes_iter)  # 取下一个字节
        val |= b & 63  # 将 b 的低 6 位加入 val
    return val  # 返回解析后的值


def check_exception_table(tab: List[ExceptionTableEntry]) -> None:
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
    for i in range(len(tab) - 1):
        assert (
            tab[i].start <= tab[i].end  # 确保起始位置小于等于结束位置
            and tab[i].end < tab[i + 1].start  # 确保当前条目的结束位置小于下一条目的起始位置
            and tab[i + 1].start <= tab[i + 1].end  # 确保下一条目的起始位置小于等于结束位置
        )


def parse_exception_table(exntab: bytes) -> List[ExceptionTableEntry]:
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    exntab_iter = iter(exntab)  # 创建字节迭代器
    tab = []  # 创建空列表用于存储 ExceptionTableEntry 对象
    try:
        while True:
            start = decode_exception_table_varint(exntab_iter) * 2  # 解析起始位置
            length = decode_exception_table_varint(exntab_iter) * 2  # 解析长度
            end = start + length - 2  # 计算结束位置
            target = decode_exception_table_varint(exntab_iter) * 2  # 解析目标位置
            dl = decode_exception_table_varint(exntab_iter)  # 解析深度和最后一条指令标志
            depth = dl >> 1  # 取深度（右移一位）
            lasti = bool(dl & 1)  # 取最后一条指令标志（取最低位）
            tab.append(ExceptionTableEntry(start, end, target, depth, lasti))  # 创建 ExceptionTableEntry 对象并加入列表
    except StopIteration:
        check_exception_table(tab)  # 校验解析后的表格是否合法
        return tab  # 返回 ExceptionTableEntry 对象的列表


def assemble_exception_table(tab: List[ExceptionTableEntry]) -> bytes:
    """
    Inverse of parse_exception_table - encodes list of exception
    table entries into bytes.
    """
    b = []  # 创建空列表用于存储字节
    for entry in tab:
        first_entry = encode_exception_table_varint(entry.start // 2)  # 编码起始位置
        first_entry[0] |= 1 << 7  # 设置第一个字节的最高位为 1
        b.extend(first_entry)  # 添加编码后的起始位置到字节列表
        length = entry.end - entry.start + 2  # 计算长度
        b.extend(encode_exception_table_varint(length // 2))  # 编码长度
        b.extend(encode_exception_table_varint(entry.target // 2))  # 编码目标位置
        dl = (entry.depth << 1) + entry.lasti  # 计算深度和最后一条指令标志
        b.extend(encode_exception_table_varint(dl))  # 编码深度和最后一条指令标志
    return bytes(b)  # 返回所有字节组成的 bytes 对象


def assemble(instructions: List[Instruction], firstlineno: int) -> Tuple[bytes, bytes]:
    """Do the opposite of dis.get_instructions()"""
    code: List[int] = []  # 创建空列表用于存储指令
    # 检查 Python 解释器版本是否大于等于 3.11
    if sys.version_info >= (3, 11):
        # 根据给定的起始行号生成行号表和更新行号的函数
        lnotab, update_lineno = linetable_311_writer(firstlineno)
        # 初始化扩展指令的数量
        num_ext = 0
        # 遍历指令列表
        for i, inst in enumerate(instructions):
            # 检查当前指令是否为 EXTENDED_ARG
            if inst.opname == "EXTENDED_ARG":
                # 设置当前指令的大小为 1
                inst_size = 1
                # 增加扩展指令的数量
                num_ext += 1
                # 从实际指令中复制位置信息
                for j in (1, 2, 3):
                    # 找到下一个非 EXTENDED_ARG 指令并复制位置信息
                    if instructions[i + j].opname != "EXTENDED_ARG":
                        inst.positions = instructions[i + j].positions
                        break
            else:
                # 计算当前指令的大小，并考虑扩展指令的影响
                inst_size = instruction_size(inst) // 2 + num_ext
                # 重置扩展指令的数量
                num_ext = 0
            
            # 更新行号表中指令位置的信息
            update_lineno(inst.positions, inst_size)
            
            # 重置扩展指令的数量
            num_ext = 0
            
            # 获取指令的参数值，默认为 0
            arg = inst.arg or 0
            
            # 将指令的操作码和参数值附加到代码列表中
            code.extend((inst.opcode, arg & 0xFF))
            
            # 补充空的字节，使得指令大小满足需求
            for _ in range(instruction_size(inst) // 2 - 1):
                code.extend((0, 0))
    else:
        # 如果 Python 版本小于 3.11，根据版本号选择行号表生成函数和更新函数
        if sys.version_info < (3, 10):
            lnotab, update_lineno = lnotab_writer(firstlineno)
        else:
            lnotab, update_lineno, end = linetable_310_writer(firstlineno)
        
        # 遍历指令列表
        for inst in instructions:
            # 如果指令包含起始行信息，更新行号
            if inst.starts_line is not None:
                update_lineno(inst.starts_line, len(code))
            
            # 获取指令的参数值，默认为 0
            arg = inst.arg or 0
            
            # 将指令的操作码和参数值附加到代码列表中
            code.extend((inst.opcode, arg & 0xFF))
        
        # 如果 Python 版本大于等于 3.10，执行结束函数
        if sys.version_info >= (3, 10):
            end(len(code))
    
    # 返回代码的字节表示和行号表的字节表示
    return bytes(code), bytes(lnotab)
def _get_instruction_by_offset(offset_to_inst: Dict[int, Instruction], offset: int):
    """
    根据给定的偏移量获取指令，考虑到 EXTENDED_ARG 指令
    """
    for n in (0, 2, 4, 6):
        # 检查当前偏移量处的指令是否不是 EXTENDED_ARG 指令，如果不是则返回该指令
        if offset_to_inst[offset + n].opcode != dis.EXTENDED_ARG:
            return offset_to_inst[offset + n]
    # 如果都是 EXTENDED_ARG 指令，则返回 None
    return None


def virtualize_jumps(instructions) -> None:
    """
    替换跳转目标，以便更容易进行编辑
    """
    # 创建跳转目标字典，键为指令的偏移量，值为指令对象本身
    jump_targets = {inst.offset: inst for inst in instructions}

    for inst in instructions:
        # 如果指令的操作码属于具有绝对地址跳转或相对地址跳转的指令集合
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            # 设置指令的目标为通过偏移量获取的目标指令
            inst.target = _get_instruction_by_offset(jump_targets, inst.argval)


_REL_JUMPS = set(dis.hasjrel)


def flip_jump_direction(instruction: Instruction) -> None:
    """
    反转跳转方向，用于特定版本的 Python
    """
    if sys.version_info < (3, 11):
        raise RuntimeError("Cannot flip jump direction in Python < 3.11")
    # 根据指令的操作名称确定其新的操作名称，用于反转跳转方向
    if "FORWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("FORWARD", "BACKWARD")
    elif "BACKWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("BACKWARD", "FORWARD")
    else:
        raise AttributeError("Instruction is not a forward or backward jump")
    # 更新指令的操作码为新的操作名称对应的操作码
    instruction.opcode = dis.opmap[instruction.opname]
    # 断言新的操作码在相对跳转集合中
    assert instruction.opcode in _REL_JUMPS


def _get_instruction_front(instructions: List[Instruction], idx: int):
    """
    获取指令前方的 EXTENDED_ARG 指令（如果有的话），用于处理跳转时的特殊情况
    """
    # 默认目标为当前索引处的指令
    target = instructions[idx]
    for offset in (1, 2, 3):
        # 如果当前索引大于等于偏移量，并且前方指令为 EXTENDED_ARG 指令，则更新目标指令
        if idx >= offset and instructions[idx - offset].opcode == dis.EXTENDED_ARG:
            target = instructions[idx - offset]
        else:
            break
    return target


def devirtualize_jumps(instructions):
    """
    在指令可能已经移动后，填充虚拟化跳转目标的参数
    """
    # 获取指令的索引信息
    indexof = get_indexof(instructions)
    # 合并具有绝对地址跳转和相对地址跳转的指令集合
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))
    # 遍历指令列表，处理每一条指令
    for inst in instructions:
        # 检查指令的操作码是否在跳转操作码集合中
        if inst.opcode in jumps:
            # 获取跳转目标指令的前导指令对象
            target = _get_instruction_front(instructions, indexof[inst.target])
            # 如果指令操作码在绝对跳转集合中
            if inst.opcode in dis.hasjabs:
                # 对于 Python 版本低于 3.10，使用目标指令的偏移量作为参数
                if sys.version_info < (3, 10):
                    inst.arg = target.offset
                # 对于 Python 版本介于 3.10 和 3.11 之间
                elif sys.version_info < (3, 11):
                    # `arg` 应为字节码偏移量，而 `offset` 是字节偏移量，因此除以 2
                    inst.arg = int(target.offset / 2)
                else:
                    # Python 3.11+ 不应存在绝对跳转
                    raise RuntimeError("Python 3.11+ should not have absolute jumps")
            else:  # 相对跳转情况
                # 计算目标和当前指令之间的字节偏移量
                inst.arg = int(target.offset - inst.offset - instruction_size(inst))
                # 如果偏移量为负数
                if inst.arg < 0:
                    # 对于 Python 版本大于等于 3.11，不允许负跳转偏移
                    if sys.version_info < (3, 11):
                        raise RuntimeError("Got negative jump offset for Python < 3.11")
                    inst.arg = -inst.arg  # 将负跳转偏移转换为正数
                    # 如果指令操作名包含 "FORWARD"，将正向跳转改为反向
                    if "FORWARD" in inst.opname:
                        flip_jump_direction(inst)
                elif inst.arg > 0:
                    # 如果偏移量为正数，且 Python 版本大于等于 3.11，并且指令操作名包含 "BACKWARD"，将反向跳转改为正向
                    if sys.version_info >= (3, 11) and "BACKWARD" in inst.opname:
                        flip_jump_direction(inst)
                # 对于 Python 版本大于等于 3.10，除以 2 是为了适应字节码大小的注释见绝对跳转情况
                if sys.version_info >= (3, 10):
                    inst.arg //= 2
            # 设置 `argval` 为目标指令的偏移量
            inst.argval = target.offset
            # 设置 `argrepr` 为格式化字符串，描述跳转至目标指令的偏移量
            inst.argrepr = f"to {target.offset}"
def virtualize_exception_table(exn_tab_bytes: bytes, instructions: List[Instruction]):
    """Replace exception table entries with pointers to make editing easier"""
    # 解析异常表字节码，转换为异常表对象列表
    exn_tab = parse_exception_table(exn_tab_bytes)
    
    # 将指令列表转换为字典，键为指令偏移量，值为指令对象，以便快速查找
    offset_to_inst = {cast(int, inst.offset): inst for inst in instructions}
    
    # 获取指令偏移量的排序列表
    offsets = sorted(offset_to_inst.keys())
    
    # 记录结束偏移量的索引，初始化为0
    end_offset_idx = 0
    
    # 异常表迭代器
    exn_tab_iter = iter(exn_tab)
    
    try:
        # 定义内部函数，用于处理异常表条目的转换
        def step():
            nonlocal end_offset_idx
            # 获取下一个异常表条目
            entry = next(exn_tab_iter)
            
            # 查找小于等于 entry.end 的最右偏移量，因为 entry.end 可能不是实际的指令偏移量
            while (end_offset_idx < len(offsets) and offsets[end_offset_idx] <= entry.end):
                end_offset_idx += 1
            
            # 断言结束偏移量索引大于0
            assert end_offset_idx > 0
            
            # 获取最近的结束偏移量
            end_offset = offsets[end_offset_idx - 1]
            
            # 构建异常表条目对象
            inst_entry = InstructionExnTabEntry(
                _get_instruction_by_offset(offset_to_inst, entry.start),
                _get_instruction_by_offset(offset_to_inst, end_offset),
                _get_instruction_by_offset(offset_to_inst, entry.target),
                entry.depth,
                entry.lasti,
            )
            return entry, inst_entry
        
        # 初始化第一个异常表条目和其对应的指令条目
        entry, inst_entry = step()
        
        # 遍历指令列表
        for inst in instructions:
            # 如果指令偏移量大于当前异常表条目的结束偏移量，则更新异常表条目和其对应的指令条目
            while inst.offset > entry.end:
                entry, inst_entry = step()
            
            # 如果指令偏移量大于等于当前异常表条目的开始偏移量，则将指令关联的异常表条目设置为复制的 inst_entry
            if inst.offset >= entry.start:
                inst.exn_tab_entry = copy.copy(inst_entry)
    
    # 处理异常表迭代器结束的情况
    except StopIteration:
        pass


def compute_exception_table(
    instructions: List[Instruction],
) -> List[ExceptionTableEntry]:
    """Compute exception table in list format from instructions with exn_tab_entries"""
    # 初始化异常表字典，键为起始和结束偏移量的元组，值为目标偏移量、深度和 lasti 值的元组
    exn_dict: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
    
    # 获取指令索引的字典
    indexof = get_indexof(instructions)

    # 遍历指令列表
    for inst in instructions:
        # 如果指令包含异常表条目
        if inst.exn_tab_entry:
            # 考虑前缀 EXTENDED_ARGS 的情况，获取起始偏移量
            start = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.start]
            ).offset
            
            # 指向结束指令的倒数第二个字节
            end = (
                cast(int, inst.exn_tab_entry.end.offset)
                + instruction_size(inst.exn_tab_entry.end)
                - 2
            )
            
            # 获取目标偏移量
            target = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.target]
            ).offset
            
            # 构建键值对，以起始和结束偏移量为键，目标偏移量、深度和 lasti 值为值
            key = (start, end)
            val = (target, inst.exn_tab_entry.depth, inst.exn_tab_entry.lasti)
            
            # 断言确保键在异常表字典中不存在或已存在且值相同
            if key in exn_dict:
                assert exn_dict[key] == val
            
            # 添加到异常表字典中
            exn_dict[key] = val
    
    # Dynamo 可能会为方便起见构造嵌套的异常表条目，但 Python 要求异常表条目不能重叠。
    # NOTE: below, "keys" refer to old instruction entries' starts and ends,
    # and "entries" refer to the generated exception table entries.

    # Sort keys by increasing start, then decreasing end
    keys_sorted = sorted(exn_dict.keys(), key=lambda t: (t[0], -t[1]))
    # smallest byte that the next exception table entry can start at
    nexti = 0
    # stack of current nested keys
    key_stack: List[Tuple[int, int]] = []
    exn_tab: List[ExceptionTableEntry] = []

    def pop():
        """
        Pop the key_stack and append an exception table entry if possible.
        """
        nonlocal nexti
        if key_stack:
            key = key_stack.pop()
            if nexti <= key[1]:
                # Append an exception table entry based on the popped key
                exn_tab.append(
                    ExceptionTableEntry(max(key[0], nexti), key[1], *exn_dict[key])
                )
                # Update nexti to the next available byte after this entry
                nexti = key[1] + 2

    for key in keys_sorted:
        # pop keys that are no longer nested over the current key
        while key_stack and key_stack[-1][1] < key[0]:
            pop()
        if key_stack:
            # create an entry covering to the current key, if possible
            assert key_stack[-1][0] <= key[0] <= key[1] <= key_stack[-1][1]
            left = max(nexti, key_stack[-1][0])
            if left < key[0]:
                # Append an exception table entry for the gap before current key
                exn_tab.append(
                    ExceptionTableEntry(left, key[0] - 2, *exn_dict[key_stack[-1]])
                )
            # Update nexti to the start of current key
            nexti = key[0]
        # Push current key onto the key_stack
        key_stack.append(key)
    # Pop any remaining keys in key_stack
    while key_stack:
        pop()
    # Validate and return the generated exception table
    check_exception_table(exn_tab)
    return exn_tab
def check_inst_exn_tab_entries_nested(
    tab: List[InstructionExnTabEntry], indexof
) -> None:
    """
    检查 `tab` 是否是正确排序的嵌套 InstructionExnTabEntry 列表，
    即没有部分重叠的条目。
    "正确排序" 意味着条目按开始递增，然后按结束递减排序。
    """
    entry_stack: List[Tuple[int, int]] = []  # 初始化空栈，用于存储 (开始索引, 结束索引) 元组
    for entry in tab:
        key = (indexof[entry.start], indexof[entry.end])
        # 处理栈中结束索引小于当前条目开始索引的情况
        while entry_stack and entry_stack[-1][1] < key[0]:
            entry_stack.pop()
        # 如果栈不为空，检查当前条目的开始索引和结束索引是否符合栈顶元素的要求
        if entry_stack:
            assert entry_stack[-1][0] <= key[0] <= key[1] <= entry_stack[-1][1]
        entry_stack.append(key)


def propagate_inst_exn_table_entries(instructions: List[Instruction]) -> None:
    """
    将异常表条目复制到其范围内所有指令中。
    支持嵌套异常表条目。
    """
    indexof = get_indexof(instructions)  # 获取指令的索引字典
    entries: Dict[Tuple[int, int], InstructionExnTabEntry] = {}
    for inst in instructions:
        if inst.exn_tab_entry:
            key = (
                indexof[inst.exn_tab_entry.start],
                indexof[inst.exn_tab_entry.end],
            )
            if key in entries:
                assert inst.exn_tab_entry == entries[key]
            entries[key] = inst.exn_tab_entry
    # 按照开始索引递增、结束索引递减的顺序排序异常表条目
    sorted_entries = [
        entries[key] for key in sorted(entries.keys(), key=lambda t: (t[0], -t[1]))
    ]
    # 检查排序后的异常表条目是否符合嵌套要求
    check_inst_exn_tab_entries_nested(sorted_entries, indexof)
    # 由于嵌套条目按排序后顺序，因此嵌套条目的传播有效。
    for entry in sorted_entries:
        # 复制条目到其范围内所有指令中
        for i in range(indexof[entry.start], indexof[entry.end] + 1):
            instructions[i].exn_tab_entry = copy.copy(entry)


def check_inst_exn_tab_entries_valid(instructions: List[Instruction]):
    """
    检查指令的异常表条目是否有效。
    条目的开始、结束和目标必须在指令中。
    带有异常表条目的指令位于条目的开始和结束指令之间。
    指令不共享异常表条目。

    隐含地检查不重复的指令。
    """
    indexof = get_indexof(instructions)  # 获取指令的索引字典
    exn_tab_entry_set = set()
    for i, inst in enumerate(instructions):
        if inst.exn_tab_entry:
            assert sys.version_info >= (3, 11)
            assert id(inst.exn_tab_entry) not in exn_tab_entry_set
            exn_tab_entry_set.add(id(inst.exn_tab_entry))
            entry = inst.exn_tab_entry
            # 检查条目的开始、结束和目标是否在索引字典中
            assert entry.start in indexof
            assert entry.end in indexof
            assert entry.target in indexof
            # 检查指令索引是否在条目的开始和结束索引之间
            assert indexof[entry.start] <= i <= indexof[entry.end]


def strip_extended_args(instructions: List[Instruction]) -> None:
    # 去除指令列表中所有操作码为 dis.EXTENDED_ARG 的指令
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


def remove_load_call_method(instructions: List[Instruction]) -> List[Instruction]:
    # 函数未提供代码，无需添加注释
    pass
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    # 检查 Python 版本是否小于 3.11，因为在这个版本之前会出现问题
    assert sys.version_info < (3, 11)
    # 定义需要重写的指令名称及其对应的新指令名称
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    # 遍历指令列表中的每一条指令
    for inst in instructions:
        # 如果当前指令的操作码在需要重写的列表中
        if inst.opname in rewrites:
            # 将当前指令的操作名称替换为新的操作名称
            inst.opname = rewrites[inst.opname]
            # 更新当前指令的操作码为对应的新操作名称的操作码
            inst.opcode = dis.opmap[inst.opname]
    # 返回更新后的指令列表
    return instructions
# 从指令列表中移除包含"_NONE"的指令
def remove_jump_if_none(instructions: List[Instruction]) -> None:
    # 创建一个新的指令列表
    new_insts = []
    # 遍历每个指令
    for inst in instructions:
        # 将当前指令添加到新的指令列表中
        new_insts.append(inst)
        # 如果指令的操作名包含"_NONE"
        if "_NONE" in inst.opname:
            # 创建一个"IS_OP"指令，根据操作名中是否包含"NOT"来设置参数
            is_op = create_instruction("IS_OP", arg=int("NOT" in inst.opname))
            is_op.argval = is_op.arg
            is_op.positions = inst.positions
            # 根据 Python 版本创建不同的跳转指令
            if sys.version_info < (3, 12):
                jump_op = create_instruction(
                    "POP_JUMP_FORWARD_IF_TRUE"
                    if "FORWARD" in inst.opname
                    else "POP_JUMP_BACKWARD_IF_TRUE",
                    target=inst.target,
                )
            else:
                jump_op = create_instruction("POP_JUMP_IF_TRUE", target=inst.target)
            jump_op.positions = inst.positions
            # 如果指令的异常表条目存在且结束指令为当前指令，则更新结束指令为跳转指令
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = jump_op
            # 复制异常表条目到新的指令
            is_op.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            jump_op.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            # 修改指令为"LOAD_CONST"，清空参数和参数值
            inst.opcode = dis.opmap["LOAD_CONST"]
            inst.opname = "LOAD_CONST"
            inst.arg = None
            inst.argval = None
            # 将新创建的指令添加到新的指令列表中
            new_insts.extend([is_op, jump_op])
    # 更新原指令列表为新的指令列表
    instructions[:] = new_insts


# 从指令列表中移除"BINARY_SLICE"和"STORE_SLICE"指令
def remove_binary_store_slice(instructions: List[Instruction]) -> None:
    # 创建一个新的指令列表
    new_insts = []
    # 遍历每个指令
    for inst in instructions:
        # 将当前指令添加到新的指令列表中
        new_insts.append(inst)
        # 如果指令的操作名为"BINARY_SLICE"或"STORE_SLICE"
        if inst.opname in ("BINARY_SLICE", "STORE_SLICE"):
            # 创建一个新的指令，将"SLICE"替换为"SUBSCR"
            subscr_inst = create_instruction(inst.opname.replace("SLICE", "SUBSCR"))
            # 如果指令的异常表条目存在且结束指令为当前指令，则更新结束指令为新的指令
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = subscr_inst
            # 复制异常表条目到新的指令
            subscr_inst.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            subscr_inst.positions = inst.positions
            # 修改指令为"BUILD_SLICE"，设置参数和参数值
            inst.opcode = dis.opmap["BUILD_SLICE"]
            inst.opname = "BUILD_SLICE"
            inst.arg = 2
            inst.argval = 2
            # 将新创建的指令添加到新的指令列表中
            new_insts.append(subscr_inst)
    # 更新原指令列表为新的指令列表
    instructions[:] = new_insts


# 将没有参数的super()转换为显式参数形式
def explicit_super(code: types.CodeType, instructions: List[Instruction]) -> None:
    """convert super() with no args into explicit arg form"""
    # 获取函数的闭包变量和自由变量
    cell_and_free = (code.co_cellvars or tuple()) + (code.co_freevars or tuple())
    # 如果函数没有参数，则不包含有效的"super()"调用，直接返回
    if not len(code.co_varnames):
        return
    # 创建一个空列表用于存储输出
    output = []
    # 遍历指令列表并获取索引和指令对象
    for idx, inst in enumerate(instructions):
        # 将当前指令添加到输出列表中
        output.append(inst)
        # 检查当前指令是否为LOAD_GLOBAL并且目标为"super"
        if inst.opname == "LOAD_GLOBAL" and inst.argval == "super":
            # 获取下一条指令
            nexti = instructions[idx + 1]
            # 检查下一条指令的参数和操作码是否符合特定条件
            if nexti.arg == 0 and (
                (sys.version_info >= (3, 12) and nexti.opname == "CALL")
                or (
                    sys.version_info >= (3, 11)
                    and sys.version_info < (3, 12)
                    and nexti.opname == "PRECALL"
                )
                or (sys.version_info < (3, 11) and nexti.opname == "CALL_FUNCTION")
            ):
                # 断言在cell_and_free中存在"__class__"，用于验证
                assert "__class__" in cell_and_free
                # 添加LOAD_DEREF指令来加载"__class__"
                output.append(create_instruction("LOAD_DEREF", argval="__class__"))
                # 获取函数的第一个参数名
                first_var = code.co_varnames[0]
                # 根据是否在cell_and_free中决定加载方式
                if first_var in cell_and_free:
                    output.append(create_instruction("LOAD_DEREF", argval=first_var))
                else:
                    output.append(create_instruction("LOAD_FAST", argval=first_var))
                # 更新nexti的参数和参数值
                nexti.arg = 2
                nexti.argval = 2
                # 如果下一条指令是PRECALL，同时更新后续的CALL指令
                if nexti.opname == "PRECALL":
                    call_inst = instructions[idx + 2]
                    call_inst.arg = 2
                    call_inst.argval = 2

    # 将指令列表更新为输出列表，完成指令的转换过程
    instructions[:] = output
# 定义函数用于修复指令列表中的扩展参数操作码
def fix_extended_args(instructions: List[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
    # 输出的指令列表，初始化为空列表
    output: List[Instruction] = []

    # 内部函数，用于可能地弹出指定数量的指令
    def maybe_pop_n(n):
        for _ in range(n):
            # 如果输出列表的最后一个指令是 EXTENDED_ARG 操作码，则弹出它
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    # 遍历输入的指令列表
    for inst in instructions:
        # 如果当前指令是 EXTENDED_ARG 操作码
        if inst.opcode == dis.EXTENDED_ARG:
            # 暂时保留此指令，以便不会缩小代码
            inst.arg = 0
        # 如果指令的参数值大于 0xFFFFFF
        elif inst.arg and inst.arg > 0xFFFFFF:
            # 可能地弹出前三个指令
            maybe_pop_n(3)
            # 添加三个新的 EXTENDED_ARG 操作码指令，分别右移参数值
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        # 如果指令的参数值大于 0xFFFF
        elif inst.arg and inst.arg > 0xFFFF:
            # 可能地弹出前两个指令
            maybe_pop_n(2)
            # 添加两个新的 EXTENDED_ARG 操作码指令，分别右移参数值
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        # 如果指令的参数值大于 0xFF
        elif inst.arg and inst.arg > 0xFF:
            # 可能地弹出前一个指令
            maybe_pop_n(1)
            # 添加一个新的 EXTENDED_ARG 操作码指令，右移参数值
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        # 将当前指令添加到输出列表中
        output.append(inst)

    # 计算输出列表长度和输入指令列表长度的差值
    added = len(output) - len(instructions)
    # 断言添加的指令数量不小于 0
    assert added >= 0
    # 将输入指令列表替换为输出列表
    instructions[:] = output
    # 返回添加的指令数量
    return added


# 定义函数用于计算指令对象的大小
def instruction_size(inst) -> int:
    import torch
    
    # 如果 Python 版本大于等于 3.11
    if sys.version_info >= (3, 11):
        # 返回指令对象大小的计算结果
        return 2 * (torch._C._dynamo.eval_frame.py_opcode_caches[inst.opcode] + 1)
    # 否则返回默认值 2
    return 2


# 定义函数用于检查指令列表中各指令的偏移量是否正确
def check_offsets(instructions) -> None:
    # 初始偏移量为 0
    offset = 0
    # 遍历指令列表中的每个指令对象
    for inst in instructions:
        # 断言当前指令的偏移量与预期偏移量相等
        assert inst.offset == offset
        # 更新预期偏移量，加上当前指令对象的大小
        offset += instruction_size(inst)


# 定义函数用于更新指令列表中各指令的偏移量
def update_offsets(instructions) -> None:
    # 初始偏移量为 0
    offset = 0
    # 遍历指令列表中的每个指令对象
    for inst in instructions:
        # 更新当前指令对象的偏移量
        inst.offset = offset
        # 更新偏移量，加上当前指令对象的大小
        offset += instruction_size(inst)


# 定义函数用于调试输出字节码比较结果的字符串表示
def debug_bytes(*args) -> str:
    # 索引范围是所有参数中最长的一个的长度
    index = range(max(map(len, args)))
    # 结果列表
    result = []
    # 遍历所有参数
    for arg in (
        [index] + list(args) + [[int(a != b) for a, b in zip(args[-1], args[-2])]]
    ):
        # 将每个参数转换为字符串并添加到结果列表
        result.append(" ".join(f"{x:03}" for x in arg))

    # 返回比较结果的字符串表示
    return "bytes mismatch\n" + "\n".join(result)


# 定义函数用于检查转换后的代码对象是否与原始代码对象的字节码和行号表相匹配
def debug_checks(code):
    """Make sure our assembler produces same bytes as we start with"""
    # 转换代码对象，并使用安全模式进行比较
    dode = transform_code_object(code, lambda x, y: None, safe=True)
    # 断言原始代码对象的字节码与转换后代码对象的字节码相等
    assert code.co_code == dode.co_code, debug_bytes(code.co_code, dode.co_code)
    # 断言原始代码对象的行号表与转换后代码对象的行号表相等
    assert code.co_lnotab == dode.co_lnotab, debug_bytes(code.co_lnotab, dode.co_lnotab)


# 初始化具有不同类型局部变量的集合
HAS_LOCAL = set(dis.haslocal)
# 初始化具有不同类型名称变量的集合
HAS_NAME = set(dis.hasname)
# 初始化具有不同类型自由变量的集合
HAS_FREE = set(dis.hasfree)
# 初始化具有不同常量的集合
HAS_CONST = set(dis.hasconst)


# 定义函数用于获取指定值在代码选项中常量列表中的索引
def get_const_index(code_options, val) -> int:
    # 遍历常量列表中的每个值及其索引
    for i, v in enumerate(code_options["co_consts"]):
        # 如果指定的值与当前值严格相等，则返回当前值的索引
        if val is v:
            return i
   `
    # 将元组 (val,) 添加到 code_options["co_consts"] 的末尾
    code_options["co_consts"] += (val,)
    # 返回 code_options["co_consts"] 的长度减去 1，表示新元素的索引
    return len(code_options["co_consts"]) - 1
def fix_vars(instructions: List[Instruction], code_options, varname_from_oparg=None):
    # 根据代码选项中的常量名列表创建名称到索引的映射
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}

    def get_name_index(name) -> int:
        try:
            idx = names[name]
        except KeyError:
            # 如果名称不存在于映射中，则将其添加到常量名列表中
            idx = names[name] = len(names)
            code_options["co_names"] = (*code_options["co_names"], name)
            assert len(code_options["co_names"]) == len(names)
        return idx

    if sys.version_info < (3, 11):
        # 在 Python 版本低于 3.11 时，确保 varname_from_oparg 为 None
        assert varname_from_oparg is None
        # 创建局部变量名到索引的映射
        varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
        # 创建自由变量名到索引的映射，合并 cellvars 和 freevars
        freenames = {
            name: idx
            for idx, name in enumerate(
                code_options["co_cellvars"] + code_options["co_freevars"]
            )
        }
    else:
        # 在 Python 版本 3.11 及以上，确保 varname_from_oparg 是可调用的
        assert callable(varname_from_oparg)
        # 使用 varname_from_oparg 函数生成所有变量名到索引的映射
        allnames = {}
        for idx in itertools.count():
            try:
                name = varname_from_oparg(idx)
                allnames[name] = idx
            except IndexError:
                break
        # 创建局部变量名到索引的映射
        varnames = {name: allnames[name] for name in code_options["co_varnames"]}
        # 创建自由变量名到索引的映射，合并 cellvars 和 freevars
        freenames = {
            name: allnames[name]
            for name in code_options["co_cellvars"] + code_options["co_freevars"]
        }
    # 遍历指令列表中的每个指令
    for i in range(len(instructions)):

        def should_compute_arg():
            # 判断是否应计算参数值，argval 优先于 arg
            return instructions[i].argval is not _NotProvided

        if instructions[i].opname == "LOAD_GLOBAL":
            # 对于 LOAD_GLOBAL 操作码，需要同时有 arg 和 argval - 参见 create_instruction 函数
            assert instructions[i].argval is not _NotProvided
            if sys.version_info >= (3, 11):
                assert instructions[i].arg is not None
                # 根据 argval 计算 arg 的值，并保留低位（get_name_index(instructions[i].argval) << 1）和强制第二位为 1（cast(int, instructions[i].arg) % 2）
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                # 直接使用 argval 的索引作为 arg
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_ATTR":
            # 对于 LOAD_ATTR 操作码，也需要同时有 arg 和 argval，类似于 LOAD_GLOBAL
            assert instructions[i].argval is not _NotProvided
            if sys.version_info >= (3, 12):
                assert instructions[i].arg is not None
                # 计算 arg 的值，方式与 LOAD_GLOBAL 相同
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                # 直接使用 argval 的索引作为 arg
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_SUPER_ATTR":
            # LOAD_SUPER_ATTR 操作码要求 arg 不为空，argval 也不能为 _NotProvided
            assert instructions[i].arg is not None
            assert instructions[i].argval is not _NotProvided
            # 计算 arg 的值，特别是要复制低位（get_name_index(instructions[i].argval) << 2）和强制第二位为 1（cast(int, instructions[i].arg) % 2），并加上 2
            instructions[i].arg = (
                (get_name_index(instructions[i].argval) << 2)
                + (cast(int, instructions[i].arg) % 2)
                + 2
            )
        elif instructions[i].opcode in HAS_LOCAL:
            # 如果指令的 opcode 在 HAS_LOCAL 中
            if should_compute_arg():
                # 如果需要计算参数值，则将参数值设置为 varnames 中对应的值
                instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            # 如果指令的 opcode 在 HAS_NAME 中
            if should_compute_arg():
                # 如果需要计算参数值，则将参数值设置为 argval 在名字索引中的值
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opcode in HAS_FREE:
            # 如果指令的 opcode 在 HAS_FREE 中
            if should_compute_arg():
                # 如果需要计算参数值，则将参数值设置为 freenames 中对应的值
                instructions[i].arg = freenames[instructions[i].argval]
        elif instructions[i].opcode in HAS_CONST:
            # 如果指令的 opcode 在 HAS_CONST 中
            # 注意：只有在 arg 未提供时才更新 argval。这假设 co_consts 的任何添加都是追加的。
            if instructions[i].arg is None:
                # 由于常量可能不可哈希，因此不能使用字典，而是使用 get_const_index 函数获取常量在 code_options 中的索引
                idx = get_const_index(code_options, instructions[i].argval)
                assert idx >= 0
                # 将指令的 arg 设置为常量在 code_options 中的索引
                instructions[i].arg = idx
# 清理指令集中有参数值的指令的参数。
# 这对于在生成的字节码中使用 dis 后的字节码很有用。
def clear_instruction_args(instructions):
    for inst in instructions:
        if (
            inst.argval is not _NotProvided
            and (
                inst.opcode in HAS_LOCAL
                or inst.opcode in HAS_NAME
                or inst.opcode in HAS_FREE
                or inst.opcode in HAS_CONST
            )
            and inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR", "LOAD_SUPER_ATTR")
        ):
            inst.arg = None


def get_code_keys() -> List[str]:
    # Python 3.11 对代码键的更改尚未完全记录。
    # 参见 https://github.com/python/cpython/blob/3.11/Objects/clinic/codeobject.c.h#L24
    # 查看新格式的信息。
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend(
        [
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
        ]
    )
    if sys.version_info >= (3, 11):
        keys.append("co_qualname")
    keys.append("co_firstlineno")
    if sys.version_info >= (3, 10):
        keys.append("co_linetable")
    else:
        keys.append("co_lnotab")
    if sys.version_info >= (3, 11):
        # 虽然没有文档，但是在 https://github.com/python/cpython/issues/84403 中引入
        keys.append("co_exceptiontable")
    keys.extend(
        [
            "co_freevars",
            "co_cellvars",
        ]
    )
    return keys


def transform_code_object(code, transformations, safe=False) -> types.CodeType:
    keys = get_code_keys()
    # 创建一个包含指定键值对应属性的字典
    code_options = {k: getattr(code, k) for k in keys}
    # 断言局部变量名列表的长度等于局部变量的个数
    assert len(code_options["co_varnames"]) == code_options["co_nlocals"]

    # 获取清理后的指令集
    instructions = cleaned_instructions(code, safe)
    # 传播行号到指令集中的每一条指令
    propagate_line_nums(instructions)

    # 对指令集应用变换函数
    transformations(instructions, code_options)
    # 清理并组装指令集，返回清理后的字节码和新的代码对象
    return clean_and_assemble_instructions(instructions, keys, code_options)[1]


def clean_and_assemble_instructions(
    instructions: List[Instruction], keys: List[str], code_options: Dict[str, Any]
) -> Tuple[List[Instruction], types.CodeType]:
    # 同时隐式检查是否有重复的指令
    check_inst_exn_tab_entries_valid(instructions)

    # 确保局部变量名列表长度与局部变量数目相等
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    varname_from_oparg = None
    if sys.version_info >= (3, 11):
        # 使用更新后名称的临时代码对象
        tmp_code = types.CodeType(*[code_options[k] for k in keys])
        varname_from_oparg = tmp_code._varname_from_oparg  # type: ignore[attr-defined]
    # 修正变量名
    fix_vars(instructions, code_options, varname_from_oparg=varname_from_oparg)

    dirty = True
    # 当还存在需要处理的情况时，循环执行以下操作
    while dirty:
        # 更新指令中的偏移量
        update_offsets(instructions)
        # 对指令进行虚拟跳转处理
        devirtualize_jumps(instructions)
        # 如果此次处理改变了偏移量，需要再次尝试处理
        dirty = bool(fix_extended_args(instructions))

    # 移除指令中的额外行号信息
    remove_extra_line_nums(instructions)
    # 将指令汇编成字节码并生成行号表
    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])
    
    # 根据 Python 版本选择行号表的字段名
    if sys.version_info < (3, 10):
        code_options["co_lnotab"] = lnotab
    else:
        code_options["co_linetable"] = lnotab

    # 将生成的字节码放入选项字典中
    code_options["co_code"] = bytecode
    # 分析指令的栈大小并放入选项字典中
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    
    # 断言选项字典中除了 "co_posonlyargcount" 以外的所有键与 set(keys) 相同
    assert set(keys) - {"co_posonlyargcount"} == set(code_options.keys()) - {
        "co_posonlyargcount"
    }
    
    # 如果 Python 版本大于等于 3.11，则生成并设置异常表
    if sys.version_info >= (3, 11):
        code_options["co_exceptiontable"] = assemble_exception_table(
            compute_exception_table(instructions)
        )

    # 返回处理后的指令集和生成的代码对象
    return instructions, types.CodeType(*[code_options[k] for k in keys])
def populate_kw_names_argval(instructions, consts):
    # 遍历指令列表
    for inst in instructions:
        # 如果指令操作码为 "KW_NAMES"
        if inst.opname == "KW_NAMES":
            # 将指令的参数值设置为常量列表中对应索引的值
            inst.argval = consts[inst.arg]

# 清理指令，返回清理后的指令列表
def cleaned_instructions(code, safe=False) -> List[Instruction]:
    # 将代码对象转换为指令列表
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    # 检查指令的偏移量
    check_offsets(instructions)
    # 如果 Python 版本大于等于 3.11
    if sys.version_info >= (3, 11):
        # 填充 KW_NAMES 操作的参数值
        populate_kw_names_argval(instructions, code.co_consts)
        # 虚拟化异常表
        virtualize_exception_table(code.co_exceptiontable, instructions)
    # 虚拟化跳转
    virtualize_jumps(instructions)
    # 去除扩展参数
    strip_extended_args(instructions)
    # 如果不安全模式
    if not safe:
        # 如果 Python 版本小于 3.11
        if sys.version_info < (3, 11):
            # 移除加载调用方法的指令
            remove_load_call_method(instructions)
        # 如果 Python 版本小于 3.12
        if sys.version_info < (3, 12):
            # 显式处理 super 关键字
            explicit_super(code, instructions)
    # 如果 Python 版本大于等于 3.11
    if sys.version_info >= (3, 11):
        # 移除跳转到 None 的指令
        remove_jump_if_none(instructions)
        # 如果 Python 版本大于等于 3.12
        if sys.version_info >= (3, 12):
            # 移除二进制存储切片的指令
            remove_binary_store_slice(instructions)
        # 更新指令的偏移量
        update_offsets(instructions)
        # 反虚拟化跳转
        devirtualize_jumps(instructions)
    # 返回处理后的指令列表
    return instructions


# 唯一标识符计数器
_unique_id_counter = itertools.count()

# 生成唯一标识符
def unique_id(name) -> str:
    return f"{name}_{next(_unique_id_counter)}"


# 判断是否为生成器函数
def is_generator(code: types.CodeType) -> bool:
    co_generator = 0x20
    return (code.co_flags & co_generator) > 0


# 从模板函数生成字节码
def bytecode_from_template(fn, varname_map=None, noreturn=True, noprefix=True):
    """Generates bytecode from a template function `fn` for use in
    dynamo bytecode generation.

    For example, we can generate Python-version-independent bytecode
    for looping through a dictionary and copying the values to a new dictionary.

    def template(d1, d2):
        for k, v in d1.items():
            d2[k] = v


    or a try block:

    def template():
        try:
            dummy1
        except:
            dummy2
            raise
        dummy3

    Args:
        fn: a function template to generate bytecode from
        varname_map: a mapping of `fn`'s varnames to new names. This
            map will be applied to the generated bytecode's varnames.
            For example, local variables in `fn` can be replaced with
            new names that are generated by `OutputGraph.new_var`.
        noreturn: remove all RETURN_* bytecodes and replace them with a jump
            to the end of the bytecode.
        noprefix: remove prefix bytecodes (all bytecode before the first RESUME, inclusive).
    """
    # 清理指令，返回清理后的指令列表
    insts = cleaned_instructions(fn.__code__)
    # 清除指令的参数
    clear_instruction_args(insts)

    # 如果需要移除前缀指令
    if noprefix:
        for i, inst in enumerate(insts):
            # 查找第一个 RESUME 指令并移除其前的所有指令
            if inst.opname == "RESUME":
                insts = insts[i + 1 :]
                break

    # 遍历指令列表
    for inst in insts:
        # 如果不重置 starts_line，生成的字节码行号将基于 fn 的行号
        inst.starts_line = None
        # 如果有变量名映射且指令的参数值在映射中
        if varname_map and inst.argval in varname_map:
            # 将参数值替换为映射中的值
            inst.argval = varname_map[inst.argval]
    # 如果设置了 noreturn 标志，执行以下操作
    if noreturn:
        # 如果 Python 版本 >= 3.12
        if sys.version_info >= (3, 12):
            # 将 RETURN_CONST 替换为 LOAD_CONST 和 RETURN_VALUE
            new_insts = []
            # 遍历指令列表中的每一个指令
            for inst in insts:
                # 如果指令的操作码是 "RETURN_CONST"
                if inst.opname == "RETURN_CONST":
                    # 修改操作码为 LOAD_CONST
                    inst.opcode = dis.opmap["LOAD_CONST"]
                    # 修改操作名为 LOAD_CONST
                    inst.opname = "LOAD_CONST"
                    # 添加修改后的指令到新的指令列表中
                    new_insts.append(inst)
                    # 添加 RETURN_VALUE 指令作为替代
                    # 不需要传播目标/异常表
                    new_insts.append(create_instruction("RETURN_VALUE"))
                else:
                    # 其他情况直接添加原指令到新指令列表中
                    new_insts.append(inst)
            # 更新指令列表为新的指令列表
            insts = new_insts

        # 初始化返回指令列表
        returns = []
        # 遍历指令列表中的每一个指令
        for inst in insts:
            # 如果指令的操作码是 "RETURN_VALUE"
            if inst.opname == "RETURN_VALUE":
                # 将该指令添加到返回指令列表中
                returns.append(inst)

        # 如果只有一个返回指令且该指令是最后一个指令
        if len(returns) == 1 and returns[0] is insts[-1]:
            # 直接弹出最后一个返回指令
            insts.pop(-1)
        elif len(returns) > 0:
            # 创建跳转目标 - 如果最后一个指令是返回指令
            if insts[-1] is returns[-1]:
                # 将最后一个返回指令替换为 NOP 并设置为跳转目标
                insts[-1].opname = "NOP"
                insts[-1].opcode = dis.opmap["NOP"]
                insts[-1].arg = None
                insts[-1].argval = _NotProvided
                returns.pop(-1)
            else:
                # 否则在指令列表末尾添加一个 NOP 指令
                insts.append(create_instruction("NOP"))

            # 将所有返回指令替换为跳转指令
            for inst in returns:
                # 为每个返回指令创建绝对跳转指令
                jump_inst = create_jump_absolute(insts[-1])
                inst.opname = jump_inst.opname
                inst.opcode = jump_inst.opcode
                inst.arg = jump_inst.arg
                inst.argval = jump_inst.argval
                inst.target = jump_inst.target

    # 返回处理后的指令列表
    return insts
```