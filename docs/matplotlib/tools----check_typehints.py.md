# `D:\src\scipysrc\matplotlib\tools\check_typehints.py`

```py
#!/usr/bin/env python
"""
Perform AST checks to validate consistency of type hints with implementation.

NOTE: in most cases ``stubtest`` (distributed as part of ``mypy``)  should be preferred

This script was written before the configuration of ``stubtest`` was well understood.
It still has some utility, particularly for checking certain deprecations or other
decorators which modify the runtime signature where you want the type hint to match
the python source rather than runtime signature, perhaps.

The basic kinds of checks performed are:

- Set of items defined by the stubs vs implementation
  - Missing stub: MISSING_STUB = 1
  - Missing implementation: MISSING_IMPL = 2
- Signatures of functions/methods
  - Positional Only Args: POS_ARGS = 4
  - Keyword or Positional Args: ARGS = 8
  - Variadic Positional Args: VARARG = 16
  - Keyword Only Args: KWARGS = 32
  - Variadic Keyword Only Args: VARKWARG = 64

There are some exceptions to when these are checked:

- Set checks (MISSING_STUB/MISSING_IMPL) only apply at the module level
  - i.e. not for classes
  - Inheritance makes the set arithmetic harder when only loading AST
  - Attributes also make it more complicated when defined in init
- Functions type hinted with ``overload`` are ignored for argument checking
  - Usually this means the implementation is less strict in signature but will raise
    if an invalid signature is used, type checking allows such errors to be caught by
    the type checker instead of at runtime.
- Private attribute/functions are ignored
  - Not expecting a type hint
  - applies to anything beginning, but not ending in ``__``
  - If ``__all__`` is defined, also applies to anything not in ``__all__``
- Deprecated methods are not checked for missing stub
  - Only applies to wholesale deprecation, not deprecation of an individual arg
  - Other kinds of deprecations (e.g. argument deletion or rename) the type hint should
    match the current python definition line still.
      - For renames, the new name is used
      - For deletions/make keyword only, it is removed upon expiry

Usage:

Currently there is not any argument handling/etc, so all configuration is done in
source.
Since stubtest has almost completely superseded this script, this is unlikely to change.

The categories outlined above can each be ignored, and ignoring multiple can be done
using the bitwise or (``|``) operator, e.g. ``ARGS | VARKWARG``.

This can be done globally or on a per file basis, by editing ``per_file_ignore``.
For the latter, the key is the Pathlib Path to the affected file, and the value is the
integer ignore.

Must be run from repository root:

``python tools/check_typehints.py``
"""

import ast
import pathlib
import sys

MISSING_STUB = 1
MISSING_IMPL = 2
POS_ARGS = 4
ARGS = 8
VARARG = 16
KWARGS = 32
VARKWARG = 64

def check_file(path, ignore=0):
    # Determine the path to the corresponding stub file by changing the extension to .pyi
    stubpath = path.with_suffix(".pyi")
    # Initialize the return value for the check
    ret = 0
    # If the stub file does not exist, return early with zero counts
    if not stubpath.exists():
        return 0, 0
    # Parse the content of the Python file at 'path' into an Abstract Syntax Tree (AST)
    tree = ast.parse(path.read_text())
    # 使用 ast 模块解析给定的 Python 源代码文件内容，生成抽象语法树
    stubtree = ast.parse(stubpath.read_text())
    # 调用函数 check_namespace，传入解析后的原始代码树 tree，解析后的存根代码树 stubtree，
    # 以及路径信息 path 和忽略列表 ignore，并返回其结果
    return check_namespace(tree, stubtree, path, ignore)
# 定义一个函数用于检查命名空间的一致性
def check_namespace(tree, stubtree, path, ignore=0):
    # 初始化返回值和计数器
    ret = 0
    count = 0

    # 从 tree 中提取所有合法的项的名称，并存储到集合 tree_items 中
    tree_items = set(
        i.name
        for i in tree.body
        if hasattr(i, "name") and (not i.name.startswith("_") or i.name.endswith("__"))
    )

    # 从 stubtree 中提取所有合法的项的名称，并存储到集合 stubtree_items 中
    stubtree_items = set(
        i.name
        for i in stubtree.body
        if hasattr(i, "name") and (not i.name.startswith("_") or i.name.endswith("__"))
    )

    # 遍历 tree 的每个项
    for item in tree.body:
        if isinstance(item, ast.Assign):
            # 如果项是赋值语句，将所有目标的名称添加到 tree_items 中
            tree_items |= set(
                i.id
                for i in item.targets
                if hasattr(i, "id")
                and (not i.id.startswith("_") or i.id.endswith("__"))
            )
            # 对于赋值语句中的元组，将每个元素的名称添加到 tree_items 中
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    tree_items |= set(i.id for i in target.elts)
        elif isinstance(item, ast.AnnAssign):
            # 如果项是注释赋值语句，将目标的名称添加到 tree_items 中
            tree_items |= {item.target.id}

    # 遍历 stubtree 的每个项
    for item in stubtree.body:
        if isinstance(item, ast.Assign):
            # 如果项是赋值语句，将所有目标的名称添加到 stubtree_items 中
            stubtree_items |= set(
                i.id
                for i in item.targets
                if hasattr(i, "id")
                and (not i.id.startswith("_") or i.id.endswith("__"))
            )
            # 对于赋值语句中的元组，将每个元素的名称添加到 stubtree_items 中
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    stubtree_items |= set(i.id for i in target.elts)
        elif isinstance(item, ast.AnnAssign):
            # 如果项是注释赋值语句，将目标的名称添加到 stubtree_items 中
            stubtree_items |= {item.target.id}

    # 尝试从 tree 中获取 "__all__" 的值并解析为列表 all_
    try:
        all_ = ast.literal_eval(ast.unparse(get_subtree(tree, "__all__").value))
    except ValueError:
        all_ = []

    # 如果 all_ 不为空，计算缺失的项
    if all_:
        missing = (tree_items - stubtree_items) & set(all_)
    else:
        missing = tree_items - stubtree_items

    # 初始化 deprecated 集合，用于存储被标记为 deprecated 的项
    deprecated = set()

    # 遍历缺失的项
    for item_name in missing:
        # 获取该项的子树
        item = get_subtree(tree, item_name)
        # 如果该项有装饰器列表，并且有 "deprecated" 装饰器，则将其加入 deprecated 集合
        if hasattr(item, "decorator_list"):
            if "deprecated" in [
                i.func.attr
                for i in item.decorator_list
                if hasattr(i, "func") and hasattr(i.func, "attr")
            ]:
                deprecated |= {item_name}

    # 如果有未被标记为 deprecated 的缺失项，并且 ignore 中未包含 MISSING_STUB 标志
    if missing - deprecated and ~ignore & MISSING_STUB:
        # 打印路径和缺失项信息，并更新 ret 和 count
        print(f"{path}: {missing - deprecated} missing from stubs")
        ret |= MISSING_STUB
        count += 1

    # 初始化 non_class_or_func 集合，用于存储不是类或函数的项
    non_class_or_func = set()

    # 遍历在 stubtree 中而不在 tree 中的项
    for item_name in stubtree_items - tree_items:
        try:
            # 尝试从 tree 中获取该项的子树，如果失败则忽略
            get_subtree(tree, item_name)
        except ValueError:
            pass
        else:
            # 如果成功获取子树，则将其加入 non_class_or_func 集合
            non_class_or_func |= {item_name}

    # 计算在 stubtree 中而不在 tree 中且不是类或函数的项
    missing_implementation = stubtree_items - tree_items - non_class_or_func

    # 如果有缺失的实现项，并且 ignore 中未包含 MISSING_IMPL 标志
    if missing_implementation and ~ignore & MISSING_IMPL:
        # 打印路径和缺失实现项信息，并更新 ret 和 count
        print(f"{path}: {missing_implementation} in stubs and not source")
        ret |= MISSING_IMPL
        count += 1
    # 对于树和存根树的交集中的每个项目名称，执行以下操作：
    for item_name in tree_items & stubtree_items:
        # 获取树中项目名称对应的子树项和存根树中项目名称对应的子树项
        item = get_subtree(tree, item_name)
        stubitem = get_subtree(stubtree, item_name)
        
        # 如果树中的项目项和存根树中的项目项都是函数定义类型
        if isinstance(item, ast.FunctionDef) and isinstance(stubitem, ast.FunctionDef):
            # 检查函数定义之间的差异，并将结果合并到 ret 中
            err, c = check_function(item, stubitem, f"{path}::{item_name}", ignore)
            ret |= err  # 将 err 按位或运算到 ret 中，更新错误标志
            count += c  # 增加计数器 c 的值
        
        # 如果树中的项目项是类定义类型
        if isinstance(item, ast.ClassDef):
            # 忽略类的集合差异... 虽然希望在初始化/方法中设置继承和属性，但缺少节点的存在和缺失均是偶发的
            # 检查命名空间中类定义的差异，并将结果合并到 ret 中
            err, c = check_namespace(
                item,
                stubitem,
                f"{path}::{item_name}",
                ignore | MISSING_STUB | MISSING_IMPL,
            )
            ret |= err  # 将 err 按位或运算到 ret 中，更新错误标志
            count += c  # 增加计数器 c 的值

    # 返回最终的错误标志 ret 和计数器 count
    return ret, count
    # 初始化返回值和计数器
    ret = 0
    count = 0

    # 检查是否在存根项的装饰器列表中找到 "overload"，如果有则假设存根项知道自己在做什么
    overloaded = "overload" in [
        i.id for i in stubitem.decorator_list if hasattr(i, "id")
    ]
    if overloaded:
        # 如果是重载函数，则直接返回 0, 0
        return 0, 0

    # 获取原函数和存根函数的位置参数列表
    item_posargs = [a.arg for a in item.args.posonlyargs]
    stubitem_posargs = [a.arg for a in stubitem.args.posonlyargs]
    # 如果位置参数列表不同，并且忽略标志中未设置 POS_ARGS 标志
    if item_posargs != stubitem_posargs and ~ignore & POS_ARGS:
        # 打印位置参数不同的信息，并更新返回值和计数器
        print(
            f"{path} {item.name} posargs differ: {item_posargs} vs {stubitem_posargs}"
        )
        ret |= POS_ARGS
        count += 1

    # 获取原函数和存根函数的普通参数列表
    item_args = [a.arg for a in item.args.args]
    stubitem_args = [a.arg for a in stubitem.args.args]
    # 如果普通参数列表不同，并且忽略标志中未设置 ARGS 标志
    if item_args != stubitem_args and ~ignore & ARGS:
        # 打印普通参数不同的信息，并更新返回值和计数器
        print(f"{path} args differ for {item.name}: {item_args} vs {stubitem_args}")
        ret |= ARGS
        count += 1

    # 获取原函数和存根函数的可变位置参数
    item_vararg = item.args.vararg
    stubitem_vararg = stubitem.args.vararg
    # 如果未设置忽略标志中的 VARARG 标志
    if ~ignore & VARARG:
        # 检查可变位置参数是否有不同
        if (item_vararg is None) ^ (stubitem_vararg is None):
            if item_vararg:
                # 如果只有一个函数有可变位置参数，则打印信息并更新返回值和计数器
                print(
                    f"{path} {item.name} vararg differ: "
                    f"{item_vararg.arg} vs {stubitem_vararg}"
                )
            else:
                print(
                    f"{path} {item.name} vararg differ: "
                    f"{item_vararg} vs {stubitem_vararg.arg}"
                )
            ret |= VARARG
            count += 1
        elif item_vararg is not None and item_vararg.arg != stubitem_vararg.arg:
            # 如果两个函数的可变位置参数不同，则打印信息并更新返回值和计数器
            print(
                f"{path} {item.name} vararg differ: "
                f"{item_vararg.arg} vs {stubitem_vararg.arg}"
            )
            ret |= VARARG
            count += 1

    # 获取原函数和存根函数的仅关键字参数列表
    item_kwonlyargs = [a.arg for a in item.args.kwonlyargs]
    stubitem_kwonlyargs = [a.arg for a in stubitem.args.kwonlyargs]
    # 如果仅关键字参数列表不同，并且忽略标志中未设置 KWARGS 标志
    if item_kwonlyargs != stubitem_kwonlyargs and ~ignore & KWARGS:
        # 打印仅关键字参数不同的信息，并更新返回值和计数器
        print(
            f"{path} {item.name} kwonlyargs differ: "
            f"{item_kwonlyargs} vs {stubitem_kwonlyargs}"
        )
        ret |= KWARGS
        count += 1

    # 获取原函数和存根函数的关键字参数
    item_kwarg = item.args.kwarg
    stubitem_kwarg = stubitem.args.kwarg
    # 检查是否需要忽略 VARKWARG 标志并且 VARKWARG 标志为真
    if ~ignore & VARKWARG:
        # 检查 item_kwarg 和 stubitem_kwarg 是否一个为 None，另一个不为 None
        if (item_kwarg is None) ^ (stubitem_kwarg is None):
            # 如果 item_kwarg 不为 None，则输出差异信息
            if item_kwarg:
                print(
                    f"{path} {item.name} varkwarg differ: "
                    f"{item_kwarg.arg} vs {stubitem_kwarg}"
                )
            # 否则，输出另一种差异信息
            else:
                print(
                    f"{path} {item.name} varkwarg differ: "
                    f"{item_kwarg} vs {stubitem_kwarg.arg}"
                )
            # 设置 ret 的 VARKWARG 标志位
            ret |= VARKWARG
            # 增加差异计数器
            count += 1
        # 如果 item_kwarg 为 None，则继续下一个判断分支
        elif item_kwarg is None:
            pass
        # 否则，比较 item_kwarg 和 stubitem_kwarg 的 arg 属性，输出差异信息
        elif item_kwarg.arg != stubitem_kwarg.arg:
            print(
                f"{path} {item.name} varkwarg differ: "
                f"{item_kwarg.arg} vs {stubitem_kwarg.arg}"
            )
            # 设置 ret 的 VARKWARG 标志位
            ret |= VARKWARG
            # 增加差异计数器
            count += 1

    # 返回 ret 和 count 变量的值
    return ret, count
# 定义函数，用于在给定抽象语法树 `tree` 中查找特定名称 `name` 的子树节点
def get_subtree(tree, name):
    # 遍历抽象语法树的所有节点
    for item in tree.body:
        # 如果节点是赋值语句（ast.Assign 类型）
        if isinstance(item, ast.Assign):
            # 检查赋值语句的目标是否包含目标名称 `name`
            if name in [i.id for i in item.targets if hasattr(i, "id")]:
                return item  # 找到匹配的赋值语句节点，返回该节点
            # 如果目标是元组（ast.Tuple 类型）
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    # 检查元组内的元素是否包含目标名称 `name`
                    if name in [i.id for i in target.elts]:
                        return item  # 找到匹配的赋值语句节点，返回该节点
        # 如果节点是注解赋值语句（ast.AnnAssign 类型）
        if isinstance(item, ast.AnnAssign):
            # 检查注解赋值语句的目标名称是否等于 `name`
            if name == item.target.id:
                return item  # 找到匹配的注解赋值语句节点，返回该节点
        # 如果节点没有 `name` 属性，继续下一个节点的检查
        if not hasattr(item, "name"):
            continue
        # 如果节点的名称等于目标名称 `name`
        if item.name == name:
            return item  # 找到匹配的节点，返回该节点
    # 如果没有找到匹配的节点，则抛出值错误异常
    raise ValueError(f"no such item {name} in tree")


if __name__ == "__main__":
    # 初始化输出和计数变量
    out = 0
    count = 0
    # 设置基础目录为 `lib/matplotlib`
    basedir = pathlib.Path("lib/matplotlib")
    # 设置特定文件需要忽略的错误字典
    per_file_ignore = {
        # 对于 `basedir / "__init__.py"` 文件，设置为 `MISSING_IMPL`
        basedir / "__init__.py": MISSING_IMPL,
        # 对于 `basedir / "ticker.py"` 文件，设置为 `VARKWARG`
        basedir / "ticker.py": VARKWARG,
        # 对于 `basedir / "layout_engine.py"` 文件，设置为 `VARKWARG`
        basedir / "layout_engine.py": VARKWARG,
    }
    # 遍历基础目录及其子目录下所有 `.py` 文件
    for f in basedir.rglob("**/*.py"):
        # 调用 `check_file` 函数检查文件 `f`，根据 `per_file_ignore` 设置忽略部分错误
        err, c = check_file(f, ignore=0 | per_file_ignore.get(f, 0))
        # 按位或操作更新 `out`，累计错误计数 `count`
        out |= err
        count += c
    # 打印空行和总错误数
    print("\n")
    print(f"{count} total errors found")
    # 退出程序，返回 `out` 作为退出码
    sys.exit(out)
```