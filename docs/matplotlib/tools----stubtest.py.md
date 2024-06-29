# `D:\src\scipysrc\matplotlib\tools\stubtest.py`

```py
# 导入 ast 模块，用于处理抽象语法树
import ast
# 导入 os 模块，提供与操作系统相关的功能
import os
# 导入 pathlib 模块，提供处理文件路径的类
import pathlib
# 导入 subprocess 模块，用于执行外部命令
import subprocess
# 导入 sys 模块，提供与 Python 解释器相关的功能
import sys
# 导入 tempfile 模块，用于创建临时文件和目录
import tempfile

# 获取当前文件的父目录的父目录，作为根路径
root = pathlib.Path(__file__).parent.parent

# 设置 lib 变量为根路径下的 "lib" 目录
lib = root / "lib"
# 设置 mpl 变量为 lib 目录下的 "matplotlib" 子目录
mpl = lib / "matplotlib"

# 定义 Visitor 类，继承自 ast.NodeVisitor 类
class Visitor(ast.NodeVisitor):
    def __init__(self, filepath, output):
        self.filepath = filepath
        # 获取相对于 lib 目录的路径部分作为上下文
        self.context = list(filepath.with_suffix("").relative_to(lib).parts)
        self.output = output

    # 处理 FunctionDef 节点的访问
    def visit_FunctionDef(self, node):
        # 检查函数装饰器列表中是否包含 "delete_parameter"
        for dec in node.decorator_list:
            if "delete_parameter" in ast.unparse(dec):
                # 获取被弃用参数的名称
                deprecated_arg = dec.args[1].value
                # 如果函数有可变参数（*args），则不处理该参数
                if (
                    node.args.vararg is not None
                    and node.args.vararg.arg == deprecated_arg
                ):
                    continue
                # 如果函数有关键字可变参数（**kwargs），则不处理该参数
                if (
                    node.args.kwarg is not None
                    and node.args.kwarg.arg == deprecated_arg
                ):
                    continue

                # 获取函数所在的层次结构
                parents = []
                if hasattr(node, "parent"):
                    parent = node.parent
                    while hasattr(parent, "parent") and not isinstance(
                        parent, ast.Module
                    ):
                        parents.insert(0, parent.name)
                        parent = parent.parent
                # 将含有弃用参数的函数名写入输出文件
                self.output.write(f"{'.'.join(self.context + parents)}.{node.name}\n")
                break

    # 处理 ClassDef 节点的访问
    def visit_ClassDef(self, node):
        # 检查类装饰器列表中是否包含 "define_aliases"
        for dec in node.decorator_list:
            if "define_aliases" in ast.unparse(dec):
                # 获取类的层次结构
                parents = []
                if hasattr(node, "parent"):
                    parent = node.parent
                    while hasattr(parent, "parent") and not isinstance(
                        parent, ast.Module
                    ):
                        parents.insert(0, parent.name)
                        parent = parent.parent
                # 解析装饰器参数，获取别名字典
                aliases = ast.literal_eval(dec.args[0])
                # 遍历别名字典中的值列表
                for substitutions in aliases.values():
                    parts = self.context + parents + [node.name]
                    # 将类方法的别名写入输出文件，使用正则表达式的格式
                    self.output.write(
                        "\n".join(
                            f"{'.'.join(parts)}.[gs]et_{a}\n" for a in substitutions
                        )
                    )
        # 递归访问类的子节点
        for child in ast.iter_child_nodes(node):
            self.visit(child)

# 创建一个临时目录，并在该目录下创建名为 "allowlist.txt" 的路径
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "allowlist.txt"
    # 使用 p 打开一个文件以进行写操作
    with p.open("wt") as f:
        # 遍历指定目录下所有的 .py 文件
        for path in mpl.glob("**/*.py"):
            # 创建 Visitor 对象 v，用于处理当前文件 path，并将结果输出到 f 中
            v = Visitor(path, f)
            # 解析当前文件 path 的抽象语法树并赋值给 tree
            tree = ast.parse(path.read_text())

            # 为抽象语法树中的每个节点设置父节点引用，以便进行回溯
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            # 对抽象语法树进行访问和处理
            v.visit(tree)
    
    # 运行 stubtest 命令，执行静态类型检查
    proc = subprocess.run(
        [
            "stubtest",
            "--mypy-config-file=pyproject.toml",
            "--allowlist=ci/mypy-stubtest-allowlist.txt",
            f"--allowlist={p}",
            "matplotlib",
        ],
        # 设置当前工作目录为 root
        cwd=root,
        # 设置运行环境变量，添加 MPLBACKEND=agg
        env=os.environ | {"MPLBACKEND": "agg"},
    )
    
    # 尝试删除临时文件 f.name
    try:
        os.unlink(f.name)
    except OSError:
        # 如果删除失败，捕获 OSError 异常并忽略
        pass
# 以给定进程的返回码作为参数，退出当前程序
sys.exit(proc.returncode)
```