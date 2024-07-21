# `.\pytorch\tools\linter\adapters\test_has_main_linter.py`

```py
#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a main block which raises an exception or
calls run_tests to ensure that the test will be run in OSS CI.

Takes ~2 minuters to run without the multiprocessing, probably overkill.
"""

from __future__ import annotations

import argparse  # 导入用于解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import multiprocessing as mp  # 导入用于多进程处理的模块
from enum import Enum  # 导入用于定义枚举类型的模块
from typing import NamedTuple  # 导入用于类型提示的 NamedTuple

import libcst as cst  # 导入 libcst 库，用于处理 Python 代码的语法树
import libcst.matchers as m  # 导入 libcst 的匹配器模块，用于匹配语法树中的模式


LINTER_CODE = "TEST_HAS_MAIN"


class HasMainVisiter(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found = False  # 初始化一个标志，用于记录是否找到了符合条件的 main 块

    def visit_Module(self, node: cst.Module) -> bool:
        name = m.Name("__name__")  # 匹配 __name__
        main = m.SimpleString('"__main__"') | m.SimpleString("'__main__'")  # 匹配 "__main__" 或 '__main__'
        run_test_call = m.Call(
            func=m.Name("run_tests") | m.Attribute(attr=m.Name("run_tests"))
        )  # 匹配调用 run_tests 函数或方法的语句
        # Distributed tests (i.e. MultiProcContinuousTest) calls `run_rank`
        # instead of `run_tests` in main
        run_rank_call = m.Call(
            func=m.Name("run_rank") | m.Attribute(attr=m.Name("run_rank"))
        )  # 匹配调用 run_rank 函数或方法的语句
        raise_block = m.Raise()  # 匹配 raise 语句

        # name == main or main == name
        if_main1 = m.Comparison(
            name,
            [m.ComparisonTarget(m.Equal(), main)],
        )  # 匹配条件判断语句 __name__ == "__main__"
        if_main2 = m.Comparison(
            main,
            [m.ComparisonTarget(m.Equal(), name)],
        )  # 匹配条件判断语句 "__main__" == __name__
        
        # 遍历模块的子节点
        for child in node.children:
            # 如果匹配到 if __name__ == "__main__" 或者 "__main__" == __name__ 的条件判断语句
            if m.matches(child, m.If(test=if_main1 | if_main2)):
                # 如果在该条件判断语句内部找到了 raise 语句、run_tests 调用或 run_rank 调用
                if m.findall(child, raise_block | run_test_call | run_rank_call):
                    self.found = True  # 标记为找到了符合要求的 main 块
                    break  # 停止继续查找

        return False


class LintSeverity(str, Enum):
    ERROR = "error"  # 错误严重性级别
    WARNING = "warning"  # 警告严重性级别
    ADVICE = "advice"  # 建议严重性级别
    DISABLED = "disabled"  # 禁用状态


class LintMessage(NamedTuple):
    path: str | None  # 文件路径或者 None
    line: int | None  # 行号或者 None
    char: int | None  # 字符位置或者 None
    code: str  # 代码块标识符
    severity: LintSeverity  # 代码块严重性级别
    name: str  # 代码块名称
    original: str | None  # 原始内容或者 None
    replacement: str | None  # 替换内容或者 None
    description: str | None  # 描述信息或者 None


def check_file(filename: str) -> list[LintMessage]:
    lint_messages = []  # 初始化 lint 消息列表
    # 使用文件名打开文件，并将其内容读取为字符串
    with open(filename) as f:
        file = f.read()
        # 创建一个 HasMainVisiter 实例
        v = HasMainVisiter()
        # 使用 cst.parse_module 解析文件内容，并调用 HasMainVisiter 的 visit 方法进行访问
        cst.parse_module(file).visit(v)
        # 如果没有找到主函数(main)，则生成错误信息并添加到 lint_messages 列表中
        if not v.found:
            message = (
                "Test files need to have a main block which either calls run_tests "
                + "(to ensure that the tests are run during OSS CI) or raises an exception "
                + "and added to the blocklist in test/run_test.py"
            )
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[no-main]",
                    original=None,
                    replacement=None,
                    description=message,
                )
            )
    # 返回 lint_messages 列表，其中包含了所有的 lint 错误信息
    return lint_messages
# 主程序入口点，定义为不返回任何结果的函数
def main() -> None:
    # 创建参数解析器对象，设置描述信息和文件前缀字符
    parser = argparse.ArgumentParser(
        description="test files should have main block linter",
        fromfile_prefix_chars="@",
    )
    # 添加位置参数，用于接收待检查的文件路径列表
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 解析命令行参数，并将结果存储在args对象中
    args = parser.parse_args()

    # 创建包含8个进程的进程池对象
    pool = mp.Pool(8)
    # 使用进程池并行地对args.filenames中的每个文件调用check_file函数，返回lint检查结果列表
    lint_messages = pool.map(check_file, args.filenames)
    # 关闭进程池，不再接受新的任务
    pool.close()
    # 阻塞直到所有进程完成
    pool.join()

    # 将lint_messages列表展开成一维列表flat_lint_messages
    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    # 遍历flat_lint_messages中的每个lint_message，将其转换为字典并打印为JSON格式，立即刷新输出流
    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


# 如果当前脚本作为主程序运行，则调用main函数
if __name__ == "__main__":
    main()
```