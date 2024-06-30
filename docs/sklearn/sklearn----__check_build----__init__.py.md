# `D:\src\scipysrc\scikit-learn\sklearn\__check_build\__init__.py`

```
# 引入操作系统模块，用于处理文件路径和目录操作
import os

# 本地安装错误信息的提示消息，当 scikit-learn 没有正确编译时显示
INPLACE_MSG = """
It appears that you are importing a local scikit-learn source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

# 标准安装错误信息的提示消息，用于提醒用户检查安装是否适合当前的 Python 版本、操作系统和平台
STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""


def raise_build_error(e):
    # 抛出一个易于理解的错误，列出目录内容以帮助邮件列表上的调试
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    # 如果当前目录为 "sklearn/__check_build"，则说明是本地安装，需要进行 "inplace build"
    if local_dir == "sklearn/__check_build":
        msg = INPLACE_MSG
    # 获取当前目录下的文件列表，用于错误信息显示
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if (i + 1) % 3:
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + "\n")
    # 抛出 ImportError 异常，显示错误信息、目录内容和适当的消息
    raise ImportError(
        """%s
___________________________________________________________________________
Contents of %s:
%s
___________________________________________________________________________
It seems that scikit-learn has not been built correctly.

If you have installed scikit-learn from source, please do not forget
to build the package before using it. For detailed instructions, see:
https://scikit-learn.org/dev/developers/advanced_installation.html#building-from-source
%s"""
        % (e, local_dir, "".join(dir_content).strip(), msg)
    )


try:
    # 尝试导入 _check_build 模块，这是 scikit-learn 内部用于检查构建情况的模块
    from ._check_build import check_build  # noqa
except ImportError as e:
    # 如果导入失败，则调用 raise_build_error 函数处理错误
    raise_build_error(e)
```