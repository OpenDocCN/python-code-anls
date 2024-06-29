# `.\numpy\tools\functions_missing_types.py`

```py
#!/usr/bin/env python
"""Find the functions in a module missing type annotations.

To use it run

./functions_missing_types.py <module>

and it will print out a list of functions in the module that don't
have types.

"""
import argparse  # 导入用于解析命令行参数的模块
import ast  # 导入用于抽象语法树操作的模块
import importlib  # 导入用于动态导入模块的模块
import os  # 导入与操作系统相关的功能

NUMPY_ROOT = os.path.dirname(os.path.join(
    os.path.abspath(__file__), "..",
))

# Technically "public" functions (they don't start with an underscore)
# that we don't want to include.
EXCLUDE_LIST = {
    "numpy": {
        # Stdlib modules in the namespace by accident
        "absolute_import",
        "division",
        "print_function",
        "warnings",
        "sys",
        "os",
        "math",
        # Accidentally public, deprecated, or shouldn't be used
        "Tester",
        "_core",
        "get_array_wrap",
        "int_asbuffer",
        "numarray",
        "oldnumeric",
        "safe_eval",
        "test",
        "typeDict",
        # Builtins
        "bool",
        "complex",
        "float",
        "int",
        "long",
        "object",
        "str",
        "unicode",
        # More standard names should be preferred
        "alltrue",  # all
        "sometrue",  # any
    }
}


class FindAttributes(ast.NodeVisitor):
    """Find top-level attributes/functions/classes in stubs files.

    Do this by walking the stubs ast. See e.g.

    https://greentreesnakes.readthedocs.io/en/latest/index.html

    for more information on working with Python's ast.

    """

    def __init__(self):
        self.attributes = set()

    def visit_FunctionDef(self, node):
        if node.name == "__getattr__":
            # Not really a module member.
            return
        self.attributes.add(node.name)
        # Do not call self.generic_visit; we are only interested in
        # top-level functions.
        return

    def visit_ClassDef(self, node):
        if not node.name.startswith("_"):
            self.attributes.add(node.name)
        return

    def visit_AnnAssign(self, node):
        self.attributes.add(node.target.id)


def find_missing(module_name):
    module_path = os.path.join(
        NUMPY_ROOT,
        module_name.replace(".", os.sep),
        "__init__.pyi",
    )

    module = importlib.import_module(module_name)
    module_attributes = {
        attribute for attribute in dir(module) if not attribute.startswith("_")
    }

    if os.path.isfile(module_path):
        with open(module_path) as f:
            tree = ast.parse(f.read())
        ast_visitor = FindAttributes()
        ast_visitor.visit(tree)
        stubs_attributes = ast_visitor.attributes
    else:
        # No stubs for this module yet.
        stubs_attributes = set()

    exclude_list = EXCLUDE_LIST.get(module_name, set())

    missing = module_attributes - stubs_attributes - exclude_list
    print("\n".join(sorted(missing)))


def main():
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
    parser.add_argument("module")  # 添加一个位置参数 'module'
    args = parser.parse_args()  # 解析命令行参数
    # 调用函数 `find_missing`，并传递参数 `args.module`
    find_missing(args.module)
# 如果当前脚本作为主程序执行（而不是作为模块被导入），则执行 main() 函数
if __name__ == "__main__":
    main()
```