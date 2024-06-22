# `.\pandas-ta\pandas_ta\custom.py`

```py
# 设置文件编码为 UTF-8
# -*- coding: utf-8 -*-

# 导入必要的模块
import importlib  # 动态导入模块的工具
import os  # 提供与操作系统交互的功能
import sys  # 提供与 Python 解释器交互的功能
import types  # 提供对 Python 类型和类的支持

# 从 os.path 模块中导入指定函数，避免命名冲突
from os.path import abspath, join, exists, basename, splitext
# 从 glob 模块中导入指定函数，用于文件匹配
from glob import glob

# 导入 pandas_ta 模块，并指定 AnalysisIndicators 类
import pandas_ta
from pandas_ta import AnalysisIndicators


def bind(function_name, function, method):
    """
    辅助函数，将自定义指标模块中定义的函数和类方法绑定到活动的 pandas_ta 实例。

    Args:
        function_name (str): 在 pandas_ta 中指标的名称
        function (fcn): 指标函数
        method (fcn): 与传递的函数对应的类方法
    """
    # 将指标函数绑定到 pandas_ta 实例
    setattr(pandas_ta, function_name, function)
    # 将类方法绑定到 AnalysisIndicators 类
    setattr(AnalysisIndicators, function_name, method)


def create_dir(path, create_categories=True, verbose=True):
    """
    辅助函数，为使用自定义指标设置合适的文件夹结构。每当想要设置新的自定义指标文件夹时，只需调用此函数一次。

    Args:
        path (str): 指标树的完整路径
        create_categories (bool): 如果为 True，则创建类别子文件夹
        verbose (bool): 如果为 True，则打印结果的详细输出
    """

    # 确保传递的目录存在/可读
    if not exists(path):
        os.makedirs(path)
        # 如果 verbose 为 True，则打印已创建主目录的消息
        if verbose:
            print(f"[i] Created main directory '{path}'.")

    # 列出目录的内容
    # dirs = glob(abspath(join(path, '*')))

    # 可选地添加任何缺少的类别子目录
    if create_categories:
        for sd in [*pandas_ta.Category]:
            d = abspath(join(path, sd))
            if not exists(d):
                os.makedirs(d)
                # 如果 verbose 为 True，则打印已创建空子目录的消息
                if verbose:
                    dirname = basename(d)
                    print(f"[i] Created an empty sub-directory '{dirname}'.")


def get_module_functions(module):
    """
    辅助函数，以字典形式获取导入模块的函数。

    Args:
        module: Python 模块

    Returns:
        dict: 模块函数映射
        {
            "func1_name": func1,
            "func2_name": func2,...
        }
    """
    module_functions = {}

    for name, item in vars(module).items():
        if isinstance(item, types.FunctionType):
            module_functions[name] = item

    return module_functions


def import_dir(path, verbose=True):
    # 确保传递的目录存在/可读
    if not exists(path):
        print(f"[X] Unable to read the directory '{path}'.")
        return

    # 列出目录的内容
    dirs = glob(abspath(join(path, "*")))

    # 遍历整个目录，导入找到的所有模块
    # 对每个目录进行遍历
    for d in dirs:
        # 获取目录的基本名称
        dirname = basename(d)

        # 仅在目录是有效的 pandas_ta 类别时才进行处理
        if dirname not in [*pandas_ta.Category]:
            # 如果启用了详细输出，则打印消息跳过非有效 pandas_ta 类别的子目录
            if verbose:
                print(f"[i] Skipping the sub-directory '{dirname}' since it's not a valid pandas_ta category.")
            continue

        # 对该类别（目录）中找到的每个模块进行处理
        for module in glob(abspath(join(path, dirname, "*.py"))):
            # 获取模块的名称（不带扩展名）
            module_name = splitext(basename(module))[0]

            # 确保提供的路径被包含在我们的 Python 路径中
            if d not in sys.path:
                sys.path.append(d)

            # （重新）加载指标模块
            module_functions = load_indicator_module(module_name)

            # 确定要绑定到 pandas_ta 的哪些模块函数
            fcn_callable = module_functions.get(module_name, None)
            fcn_method_callable = module_functions.get(f"{module_name}_method", None)

            # 如果找不到可调用的函数，则打印错误消息并继续下一个模块
            if fcn_callable == None:
                print(f"[X] Unable to find a function named '{module_name}' in the module '{module_name}.py'.")
                continue
            # 如果找不到可调用的方法函数，则打印错误消息并继续下一个模块
            if fcn_method_callable == None:
                missing_method = f"{module_name}_method"
                print(f"[X] Unable to find a method function named '{missing_method}' in the module '{module_name}.py'.")
                continue

            # 如果模块名称尚未在相应类别中，则将其添加到类别中
            if module_name not in pandas_ta.Category[dirname]:
                pandas_ta.Category[dirname].append(module_name)

            # 将函数绑定到 pandas_ta
            bind(module_name, fcn_callable, fcn_method_callable)
            # 如果启用了详细输出，则打印成功导入自定义指标的消息
            if verbose:
                print(f"[i] Successfully imported the custom indicator '{module}' into category '{dirname}'.")
# 将 import_dir 函数的文档字符串赋值给 import_dir.__doc__，用于说明该函数的作用和用法
import_dir.__doc__ = \
"""
Import a directory of custom indicators into pandas_ta

Args:
    path (str): Full path to your indicator tree  # 参数：指定自定义指标所在目录的完整路径
    verbose (bool): If True verbose output of results  # 参数：如果为 True，则输出详细的结果信息

This method allows you to experiment and develop your own technical analysis
indicators in a separate local directory of your choice but use them seamlessly
together with the existing pandas_ta functions just like if they were part of
pandas_ta.

If you at some late point would like to push them into the pandas_ta library
you can do so very easily by following the step by step instruction here
https://github.com/twopirllc/pandas-ta/issues/355.

A brief example of usage:

1. Loading the 'ta' module:
>>> import pandas as pd
>>> import pandas_ta as ta

2. Create an empty directory on your machine where you want to work with your
indicators. Invoke pandas_ta.custom.import_dir once to pre-populate it with
sub-folders for all available indicator categories, e.g.:

>>> import os
>>> from os.path import abspath, join, expanduser
>>> from pandas_ta.custom import create_dir, import_dir
>>> ta_dir = abspath(join(expanduser("~"), "my_indicators"))
>>> create_dir(ta_dir)

3. You can now create your own custom indicator e.g. by copying existing
ones from pandas_ta core module and modifying them.

IMPORTANT: Each custom indicator should have a unique name and have both
a) a function named exactly as the module, e.g. 'ni' if the module is ni.py
b) a matching method used by AnalysisIndicators named as the module but
   ending with '_method'. E.g. 'ni_method'

In essence these modules should look exactly like the standard indicators
available in categories under the pandas_ta-folder. The only difference will
be an addition of a matching class method.

For an example of the correct structure, look at the example ni.py in the
examples folder.

The ni.py indicator is a trend indicator so therefore we drop it into the
sub-folder named trend. Thus we have a folder structure like this:

~/my_indicators/
│
├── candles/
.
.
└── trend/
.      └── ni.py
.
└── volume/

4. We can now dynamically load all our custom indicators located in our
designated indicators directory like this:

>>> import_dir(ta_dir)

If your custom indicator(s) loaded succesfully then it should behave exactly
like all other native indicators in pandas_ta, including help functions.
"""


def load_indicator_module(name):
    """
     Helper function to (re)load an indicator module.

    Returns:
        dict: module functions mapping
        {
            "func1_name": func1,
            "func2_name": func2,...
        }

    """
    # 加载指标模块
    try:
        module = importlib.import_module(name)
    except Exception as ex:  # 捕获异常，如果加载模块出错则打印错误信息并退出程序
        print(f"[X] An error occurred when attempting to load module {name}: {ex}")
        sys.exit(1)

    # 刷新之前加载的模块，以便重新加载
    module = importlib.reload(module)
    # 返回模块函数的字典映射，包括模块中定义的所有函数
    return get_module_functions(module)
```  
```