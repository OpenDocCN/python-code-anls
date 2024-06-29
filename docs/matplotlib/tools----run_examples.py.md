# `D:\src\scipysrc\matplotlib\tools\run_examples.py`

```py
"""
Run and time some or all examples.
"""

# 导入所需模块
from argparse import ArgumentParser             # 从argparse模块导入ArgumentParser类，用于解析命令行参数
from contextlib import ExitStack                # 从contextlib模块导入ExitStack类，用于管理一组上下文管理器
import os                                       # 导入os模块，提供与操作系统交互的功能
from pathlib import Path                        # 从pathlib模块导入Path类，用于处理文件路径
import subprocess                               # 导入subprocess模块，用于执行外部命令
import sys                                      # 导入sys模块，提供对Python运行时环境的访问
from tempfile import TemporaryDirectory         # 从tempfile模块导入TemporaryDirectory类，用于创建临时目录
import time                                     # 导入time模块，提供时间相关的功能
import tokenize                                 # 导入tokenize模块，用于解析Python源代码

_preamble = """\
from matplotlib import pyplot as plt

def pseudo_show(block=True):
    for num in plt.get_fignums():
        plt.figure(num).savefig(f"{num}")

plt.show = pseudo_show

"""

# 定义RunInfo类，用于存储运行信息
class RunInfo:
    def __init__(self, backend, elapsed, failed):
        self.backend = backend                   # 后端名称
        self.elapsed = elapsed                   # 运行时间
        self.failed = failed                     # 是否运行失败

    def __str__(self):
        s = ""
        if self.backend:
            s += f"{self.backend}: "             # 如果有后端名称，将其添加到输出字符串中
        s += f"{self.elapsed}ms"                 # 添加运行时间到输出字符串中
        if self.failed:
            s += " (failed!)"                     # 如果运行失败，添加失败信息到输出字符串中
        return s                                # 返回格式化后的字符串表示


def main():
    parser = ArgumentParser(description=__doc__)  # 创建参数解析器，使用脚本的文档字符串作为描述信息
    parser.add_argument(
        "--backend", action="append",            # 添加--backend参数，可以多次使用，用于指定测试的后端
        help=("backend to test; can be passed multiple times; defaults to the "
              "default backend"))
    parser.add_argument(
        "--include-sgskip", action="store_true", # 添加--include-sgskip参数，用于指定是否包含*_sgskip.py的例子
        help="do not filter out *_sgskip.py examples")
    parser.add_argument(
        "--rundir", type=Path,                  # 添加--rundir参数，用于指定运行测试的目录
        help=("directory from where the tests are run; defaults to a "
              "temporary directory"))
    parser.add_argument(
        "paths", nargs="*", type=Path,           # 添加位置参数paths，用于指定要运行的示例文件路径
        help="examples to run; defaults to all examples (except *_sgskip.py)")
    args = parser.parse_args()                  # 解析命令行参数

    root = Path(__file__).resolve().parent.parent / "examples"  # 设置根目录为当前脚本的上两级目录下的examples目录
    paths = args.paths if args.paths else sorted(root.glob("**/*.py"))  # 如果指定了paths参数，则使用指定的文件路径，否则列出所有*.py文件
    if not args.include_sgskip:
        paths = [path for path in paths if not path.stem.endswith("sgskip")]  # 如果未指定--include-sgskip参数，则过滤掉以sgskip结尾的文件
    relpaths = [path.resolve().relative_to(root) for path in paths]  # 获取相对于root目录的所有文件路径
    width = max(len(str(relpath)) for relpath in relpaths)  # 计算文件路径字符串的最大长度
    for relpath in relpaths:
        print(str(relpath).ljust(width + 1), end="", flush=True)  # 输出对齐的文件路径，确保对齐格式
        runinfos = []                           # 存储每次运行的信息列表
        with ExitStack() as stack:
            if args.rundir:
                cwd = args.rundir / relpath.with_suffix("")  # 如果指定了--rundir参数，则运行目录为指定目录下的文件路径
                cwd.mkdir(parents=True)         # 创建运行目录，如果不存在则递归创建
            else:
                cwd = stack.enter_context(TemporaryDirectory())  # 否则使用临时目录作为运行目录
            with tokenize.open(root / relpath) as src:
                Path(cwd, relpath.name).write_text(
                    _preamble + src.read(), encoding="utf-8")  # 将源文件与_preamble内容写入到运行目录下
            for backend in args.backend or [None]:  # 遍历所有后端，如果未指定则遍历None
                env = {**os.environ}            # 复制当前环境变量
                if backend is not None:
                    env["MPLBACKEND"] = backend  # 设置MPLBACKEND环境变量为当前后端名称
                start = time.perf_counter()     # 记录开始时间
                proc = subprocess.run([sys.executable, relpath.name],
                                      cwd=cwd, env=env)  # 执行子进程运行示例文件
                elapsed = round(1000 * (time.perf_counter() - start))  # 计算运行时间并四舍五入为毫秒
                runinfos.append(RunInfo(backend, elapsed, proc.returncode))  # 将运行信息存入runinfos列表
        print("\t".join(map(str, runinfos)))    # 打印每个示例的运行信息


if __name__ == "__main__":
    main()
```