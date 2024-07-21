# `.\pytorch\torch\_dynamo\bytecode_analysis.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import bisect                  # 提供用于操作有序列表的函数
import dataclasses            # 支持用于声明数据类的装饰器
import dis                      # 支持解析和分析 Python 字节码
import sys                      # 提供与 Python 解释器交互的功能
from typing import Any, Set, Union  # 提供类型提示

# 定义终止操作码集合，表示指令执行终止的操作码集合
TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],     # 返回值操作码
    dis.opmap["JUMP_FORWARD"],     # 向前跳转操作码
    dis.opmap["RAISE_VARARGS"],    # 引发异常操作码
    # TODO(jansel): double check exception handling  # 待办事项，双重检查异常处理
}
# 根据 Python 版本，动态添加终止操作码
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap["RERAISE"])   # 重新引发异常操作码
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD"])   # 向后跳转操作码
    TERMINAL_OPCODES.add(dis.opmap["JUMP_FORWARD"])    # 向前跳转操作码
else:
    TERMINAL_OPCODES.add(dis.opmap["JUMP_ABSOLUTE"])   # 绝对跳转操作码
if sys.version_info >= (3, 12):
    TERMINAL_OPCODES.add(dis.opmap["RETURN_CONST"])    # 返回常量操作码

# 定义跳转操作码集合，表示具有跳转功能的操作码集合
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
# 定义跳转操作码名称集合，表示跳转操作码对应的操作名称集合
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
# 定义具有本地变量的操作码集合
HASLOCAL = set(dis.haslocal)
# 定义具有自由变量的操作码集合
HASFREE = set(dis.hasfree)

# 获取字节码的堆栈效果函数
stack_effect = dis.stack_effect


def get_indexof(insts):
    """
    获取指令内存地址到指令列表索引的映射。
    同时检查每个指令在列表中只出现一次。
    """
    indexof = {}
    for i, inst in enumerate(insts):
        assert inst not in indexof    # 确保每个指令在映射中只出现一次
        indexof[inst] = i             # 将指令和其索引存入映射
    return indexof


def remove_dead_code(instructions):
    """消除死代码"""
    indexof = get_indexof(instructions)   # 获取指令索引映射
    live_code = set()                    # 存储存活指令的集合

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)             # 将当前指令索引添加到存活集合中
            inst = instructions[i]
            if inst.exn_tab_entry:       # 如果指令有异常表项
                find_live_code(indexof[inst.exn_tab_entry.target])   # 递归查找目标指令的存活代码
            if inst.opcode in JUMP_OPCODES:    # 如果指令是跳转操作码
                find_live_code(indexof[inst.target])   # 递归查找目标指令的存活代码
            if inst.opcode in TERMINAL_OPCODES:    # 如果指令是终止操作码
                return    # 返回，表示找到终止代码

    find_live_code(0)   # 从第一条指令开始查找存活代码

    # 如果 Python 版本支持至少 3.11 版本
    if sys.version_info >= (3, 11):
        live_idx = sorted(live_code)   # 对存活指令索引进行排序
        for i, inst in enumerate(instructions):
            if i in live_code and inst.exn_tab_entry:
                # 查找左边最近的存活指令 >= 异常表项的起始指令
                start_idx = bisect.bisect_left(
                    live_idx, indexof[inst.exn_tab_entry.start]
                )
                assert start_idx < len(live_idx)
                # 查找右边最近的存活指令 <= 异常表项的结束指令
                end_idx = (
                    bisect.bisect_right(live_idx, indexof[inst.exn_tab_entry.end]) - 1
                )
                assert end_idx >= 0
                assert live_idx[start_idx] <= i <= live_idx[end_idx]
                inst.exn_tab_entry.start = instructions[live_idx[start_idx]]
                inst.exn_tab_entry.end = instructions[live_idx[end_idx]]
    # 返回一个列表，其中包含 instructions 列表中索引为 live_code 列表中元素的指令
    return [inst for i, inst in enumerate(instructions) if i in live_code]
# 消除跳转到下一条指令的无意义跳转
def remove_pointless_jumps(instructions):
    # 创建集合，存储所有无意义跳转指令的对象标识符
    pointless_jumps = {
        id(a)
        for a, b in zip(instructions, instructions[1:])  # 使用zip函数对指令列表进行迭代
        if a.opname == "JUMP_ABSOLUTE" and a.target is b  # 条件判断：指令为JUMP_ABSOLUTE且目标是下一条指令
    }
    # 返回移除无意义跳转后的指令列表
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


# 确保每条指令都设置了行号，以防某些指令被移除
def propagate_line_nums(instructions):
    cur_line_no = None  # 当前行号变量

    def populate_line_num(inst):
        nonlocal cur_line_no  # 使用nonlocal声明外部变量
        if inst.starts_line:
            cur_line_no = inst.starts_line  # 更新当前行号为指令的起始行号

        inst.starts_line = cur_line_no  # 设置指令的起始行号

    # 遍历所有指令，为每条指令设置行号
    for inst in instructions:
        populate_line_num(inst)


# 移除字节码打包前多余的行号属性
def remove_extra_line_nums(instructions):
    cur_line_no = None  # 当前行号变量

    def remove_line_num(inst):
        nonlocal cur_line_no  # 使用nonlocal声明外部变量
        if inst.starts_line is None:
            return  # 如果指令的起始行号为None，则直接返回
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None  # 如果指令的起始行号与当前行号相同，将其设置为None
        else:
            cur_line_no = inst.starts_line  # 更新当前行号为指令的起始行号

    # 遍历所有指令，移除多余的行号属性
    for inst in instructions:
        remove_line_num(inst)


# 执行活跃变量分析，返回必须读取和可能读取的变量集合
def livevars_analysis(instructions, instruction):
    indexof = get_indexof(instructions)  # 获取指令在列表中的索引映射
    must = ReadsWrites(set(), set(), set())  # 初始化必须读取的变量集合
    may = ReadsWrites(set(), set(), set())   # 初始化可能读取的变量集合

    def walk(state, start):
        if start in state.visited:
            return  # 如果起始指令已经被访问过，则直接返回
        state.visited.add(start)  # 将起始指令标记为已访问

        for i in range(start, len(instructions)):
            inst = instructions[i]  # 获取当前指令
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:  # 如果指令涉及局部变量或自由变量
                if "LOAD" in inst.opname or "DELETE" in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)  # 添加读取变量到相应集合
                elif "STORE" in inst.opname:
                    state.writes.add(inst.argval)  # 添加写入变量到相应集合
                elif inst.opname == "MAKE_CELL":
                    pass
                else:
                    raise NotImplementedError(f"unhandled {inst.opname}")  # 抛出未实现错误
            if inst.exn_tab_entry:
                walk(may, indexof[inst.exn_tab_entry.target])  # 递归处理异常目标指令
            if inst.opcode in JUMP_OPCODES:
                walk(may, indexof[inst.target])  # 递归处理跳转目标指令
                state = may  # 更新当前状态为可能读取状态
            if inst.opcode in TERMINAL_OPCODES:
                return  # 如果指令为终止指令，则直接返回

    walk(must, indexof[instruction])  # 从指定指令开始执行递归遍历
    return must.reads | may.reads  # 返回必须读取和可能读取的变量集合的并集


# 固定点标志类，用于存储布尔值
@dataclasses.dataclass
class FixedPointBox:
    value: bool = True  # 初始化固定点值为True


# 栈大小类，包含低位和高位的值，以及固定点标志
@dataclasses.dataclass
class StackSize:
    low: Union[int, float]
    high: Union[int, float]
    fixed_point: FixedPointBox

    def zero(self):
        self.low = 0  # 将低位值设置为0
        self.high = 0  # 将高位值设置为0
        self.fixed_point.value = False  # 将固定点标志设置为False
    # 记录调用前的自身和其他对象的低高值
    prior = (self.low, self.high)
    # 更新自身的低值，确保不小于其他对象的低值加上偏移量 n
    self.low = min(self.low, other.low + n)
    # 更新自身的高值，确保不小于其他对象的高值加上偏移量 n
    self.high = max(self.high, other.high + n)
    # 如果更新后的低高值与调用前不同，将固定点标志设置为 False
    if (self.low, self.high) != prior:
        self.fixed_point.value = False

    # 记录调用前的自身的低高值
    prior = (self.low, self.high)
    # 更新自身的低值，确保不小于给定深度 depth
    self.low = min(self.low, depth)
    # 更新自身的高值，确保不小于给定深度 depth
    self.high = max(self.high, depth)
    # 如果更新后的低高值与调用前不同，将固定点标志设置为 False
    if (self.low, self.high) != prior:
        self.fixed_point.value = False
# 分析指令集的堆栈大小变化，返回最大堆栈深度或浮点数
def stacksize_analysis(instructions) -> Union[int, float]:
    # 确保指令集非空
    assert instructions
    # 创建固定点对象
    fixed_point = FixedPointBox()
    # 初始化指令集中每个指令的堆栈大小字典，初始范围为无穷大到负无穷，使用固定点对象
    stack_sizes = {
        inst: StackSize(float("inf"), float("-inf"), fixed_point)
        for inst in instructions
    }
    # 将第一个指令的堆栈大小设置为零
    stack_sizes[instructions[0]].zero()

    # 进行最多100次的循环分析
    for _ in range(100):
        # 如果固定点值为真，则退出循环
        if fixed_point.value:
            break
        # 将固定点值设为真
        fixed_point.value = True

        # 遍历指令集及其后续指令（最后一条指令的后续指令为None）
        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            # 获取当前指令的堆栈大小对象
            stack_size = stack_sizes[inst]
            # 若当前指令是 CALL_FINALLY 并且 Python 版本低于 3.9，则设置为真
            is_call_finally = (
                sys.version_info < (3, 9) and inst.opcode == dis.opmap["CALL_FINALLY"]
            )
            # 如果当前指令不是终止指令
            if inst.opcode not in TERMINAL_OPCODES:
                # 确保存在下一条指令，否则抛出异常
                assert next_inst is not None, f"missing next inst: {inst}"
                # CALL_FINALLY 和 END_FINALLY 在 3.8 中的总堆栈影响为 0
                eff = (
                    0
                    if is_call_finally
                    else stack_effect(inst.opcode, inst.arg, jump=False)
                )
                # 将当前指令的堆栈影响传递给下一条指令
                stack_sizes[next_inst].offset_of(stack_size, eff)
            # 如果当前指令是跳转指令且不是 CALL_FINALLY
            if inst.opcode in JUMP_OPCODES and not is_call_finally:
                # 将跳转目标的堆栈影响传递给当前指令的目标
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True)
                )
            # 如果当前指令存在异常表条目
            if inst.exn_tab_entry:
                # 根据 Python 3.11 的异常处理文档，计算深度
                depth = inst.exn_tab_entry.depth + int(inst.exn_tab_entry.lasti) + 1
                # 更新异常表条目对应目标的堆栈大小
                stack_sizes[inst.exn_tab_entry.target].exn_tab_jump(depth)

    # 如果条件为假，则打印每个指令的堆栈大小信息
    if False:
        for inst in instructions:
            stack_size = stack_sizes[inst]
            print(stack_size.low, stack_size.high, inst)

    # 计算堆栈大小字典中最小和最大的低和高值
    low = min(x.low for x in stack_sizes.values())
    high = max(x.high for x in stack_sizes.values())

    # 确保达到了固定点
    assert fixed_point.value, "failed to reach fixed point"
    # 确保最小堆栈深度大于等于0
    assert low >= 0
    # 返回最大堆栈深度
    return high
```