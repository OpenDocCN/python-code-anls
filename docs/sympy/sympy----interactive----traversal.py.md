# `D:\src\scipysrc\sympy\sympy\interactive\traversal.py`

```
# 导入 sympy 库中的 Basic 类，用于处理数学表达式的基本操作
# 导入 pprint 函数，用于美观地打印输出数学表达式

from sympy.core.basic import Basic
from sympy.printing import pprint

# 导入 random 库，用于生成随机数

import random

# 定义函数 interactive_traversal，用于交互式地遍历树形结构的表达式
def interactive_traversal(expr):
    """Traverse a tree asking a user which branch to choose. """

    # ANSI 控制字符，用于控制终端文本颜色
    RED, BRED = '\033[0;31m', '\033[1;31m'         # 红色文本，常规和粗体
    GREEN, BGREEN = '\033[0;32m', '\033[1;32m'     # 绿色文本，常规和粗体
    YELLOW, BYELLOW = '\033[0;33m', '\033[1;33m'   # 黄色文本，常规和粗体
    BLUE, BBLUE = '\033[0;34m', '\033[1;34m'       # 蓝色文本，常规和粗体
    MAGENTA, BMAGENTA = '\033[0;35m', '\033[1;35m' # 紫红色文本，常规和粗体
    CYAN, BCYAN = '\033[0;36m', '\033[1;36m'       # 青色文本，常规和粗体
    END = '\033[0m'                                # 恢复默认文本颜色

    # 定义 cprint 函数，用于打印带有 ANSI 控制字符的彩色文本
    def cprint(*args):
        print("".join(map(str, args)) + END)
    # 定义一个名为 _interactive_traversal 的函数，用于交互式遍历表达式树
    def _interactive_traversal(expr, stage):
        # 如果遍历阶段大于0，则打印空行
        if stage > 0:
            print()

        # 使用 cprint 函数打印当前表达式的阶段信息
        cprint("Current expression (stage ", BYELLOW, stage, END, "):")
        # 打印表达式的表示
        print(BCYAN)
        pprint(expr)
        print(END)

        # 根据表达式的类型进行处理
        if isinstance(expr, Basic):
            # 如果是加法表达式，则按顺序获取项
            if expr.is_Add:
                args = expr.as_ordered_terms()
            # 如果是乘法表达式，则按顺序获取因子
            elif expr.is_Mul:
                args = expr.as_ordered_factors()
            # 否则直接获取表达式的参数
            else:
                args = expr.args
        # 如果表达式可迭代，则转换为列表
        elif hasattr(expr, "__iter__"):
            args = list(expr)
        # 否则直接返回表达式
        else:
            return expr

        # 获取参数列表的长度
        n_args = len(args)

        # 如果参数列表为空，则直接返回表达式
        if not n_args:
            return expr

        # 遍历参数列表，打印每个参数的类型和内容
        for i, arg in enumerate(args):
            cprint(GREEN, "[", BGREEN, i, GREEN, "] ", BLUE, type(arg), END)
            pprint(arg)
            print()

        # 根据参数个数设定用户选择范围
        if n_args == 1:
            choices = '0'
        else:
            choices = '0-%d' % (n_args - 1)

        try:
            # 获取用户输入的选择
            choice = input("Your choice [%s,f,l,r,d,?]: " % choices)
        except EOFError:
            # 如果发生 EOFError，则返回当前表达式
            result = expr
            print()
        else:
            # 根据用户选择进行不同的操作
            if choice == '?':
                # 打印帮助信息并递归调用 _interactive_traversal 函数
                cprint(RED, "%s - select subexpression with the given index" %
                       choices)
                cprint(RED, "f - select the first subexpression")
                cprint(RED, "l - select the last subexpression")
                cprint(RED, "r - select a random subexpression")
                cprint(RED, "d - done\n")

                result = _interactive_traversal(expr, stage)
            elif choice in ('d', ''):
                # 如果选择 'd' 或为空，则返回当前表达式
                result = expr
            elif choice == 'f':
                # 选择 'f'，则递归调用 _interactive_traversal 处理第一个子表达式
                result = _interactive_traversal(args[0], stage + 1)
            elif choice == 'l':
                # 选择 'l'，则递归调用 _interactive_traversal 处理最后一个子表达式
                result = _interactive_traversal(args[-1], stage + 1)
            elif choice == 'r':
                # 选择 'r'，则随机选择一个子表达式进行递归处理
                result = _interactive_traversal(random.choice(args), stage + 1)
            else:
                try:
                    # 尝试将选择转换为整数
                    choice = int(choice)
                except ValueError:
                    # 若转换失败，则打印错误信息并重新递归调用处理
                    cprint(BRED,
                           "Choice must be a number in %s range\n" % choices)
                    result = _interactive_traversal(expr, stage)
                else:
                    # 如果选择合法，则递归处理对应的子表达式
                    if choice < 0 or choice >= n_args:
                        cprint(BRED, "Choice must be in %s range\n" % choices)
                        result = _interactive_traversal(expr, stage)
                    else:
                        result = _interactive_traversal(args[choice], stage + 1)

        # 返回处理后的结果
        return result

    # 返回对表达式进行交互式遍历后的最终结果
    return _interactive_traversal(expr, 0)
```