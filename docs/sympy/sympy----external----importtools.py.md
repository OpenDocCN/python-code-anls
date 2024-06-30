# `D:\src\scipysrc\sympy\sympy\external\importtools.py`

```
# 辅助工具来帮助导入可选的外部模块

import sys  # 导入sys模块，提供对Python解释器的访问
import re   # 导入re模块，提供正则表达式支持

# 在模块中覆盖这些变量以改变默认的警告行为。
# 例如，在运行测试之前，您可以将它们都设置为False，以便警告不会打印到控制台上，或者设置为True以进行调试。
WARN_NOT_INSTALLED = None  # 默认为False
WARN_OLD_VERSION = None    # 默认为True


def __sympy_debug():
    # 从sympy/__init__.py中导入的辅助函数
    # 我们不直接从该文件导入SYMPY_DEBUG，因为我们不想导入整个SymPy来使用这个模块。
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')  # 从环境变量SYMPY_DEBUG中获取调试标志，默认为'False'
    if debug_str in ('True', 'False'):
        return eval(debug_str)  # 将调试标志字符串转换为布尔值并返回
    else:
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" % debug_str)  # 如果调试标志不是'True'或'False'，则引发运行时错误


if __sympy_debug():
    WARN_OLD_VERSION = True   # 如果处于调试模式，则将WARN_OLD_VERSION设置为True
    WARN_NOT_INSTALLED = True  # 如果处于调试模式，则将WARN_NOT_INSTALLED设置为True


_component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)

def version_tuple(vstring):
    # 将版本字符串解析为元组，例如 '1.2' -> (1, 2)
    # 简化自distutils.version.LooseVersion，该模块在Python 3.10中已被弃用。
    components = []
    for x in _component_re.split(vstring):  # 使用预编译的正则表达式_component_re来拆分版本字符串
        if x and x != '.':  # 如果拆分结果非空且不为'.'
            try:
                x = int(x)  # 尝试将拆分结果转换为整数
            except ValueError:
                pass  # 如果转换失败则跳过
            components.append(x)  # 将处理后的部分添加到components列表中
    return tuple(components)  # 返回组成的元组


def import_module(module, min_module_version=None, min_python_version=None,
        warn_not_installed=None, warn_old_version=None,
        module_version_attr='__version__', module_version_attr_call_args=None,
        import_kwargs={}, catch=()):
    """
    导入并返回一个已安装的模块。

    如果模块未安装，则返回None。

    可以通过关键字参数min_module_version指定模块的最低版本。这应与模块版本进行比较。
    默认情况下，使用module.__version__来获取模块的版本。要覆盖此行为，请设置module_version_attr关键字参数。
    如果需要调用模块属性以获取版本（例如，module.version()），则设置module_version_attr_call_args以便module.module_version_attr(*module_version_attr_call_args)返回模块的版本。

    如果模块版本小于min_module_version（使用Python的<比较），即使模块已安装，也将返回None。可以使用此功能防止导入不兼容的旧版本模块。

    还可以通过min_python_version关键字参数指定最低Python版本。这应与sys.version_info进行比较。

    如果设置关键字参数warn_not_installed为True，则在模块未安装时函数将发出UserWarning。

    如果设置关键字参数warn_old_version为True，则函数将

    """

    # 函数体未完整列出，请参考实际代码实现
    """
    # 如果 WARN_OLD_VERSION 未定义，则使用 warn_old_version 或默认为 True
    warn_old_version = (WARN_OLD_VERSION if WARN_OLD_VERSION is not None
        else warn_old_version or True)
    # 如果 WARN_NOT_INSTALLED 未定义，则使用 warn_not_installed 或默认为 False
    warn_not_installed = (WARN_NOT_INSTALLED if WARN_NOT_INSTALLED is not None
        else warn_not_installed or False)

    # 导入警告模块
    import warnings

    # 首先检查 Python 版本，以避免导入无法使用的模块
    if min_python_version:
        # 如果当前 Python 版本低于指定的最低版本
        if sys.version_info < min_python_version:
            # 如果设置了 warn_old_version 并且为 True，则发出警告
            if warn_old_version:
                warnings.warn("Python version is too old to use %s "
                    "(%s or newer required)" % (
                        module, '.'.join(map(str, min_python_version))),
                    UserWarning, stacklevel=2)
            # 返回，不继续导入模块
            return
    """
    try:
        # 尝试动态导入指定模块，使用传入的关键字参数
        mod = __import__(module, **import_kwargs)

        ## there's something funny about imports with matplotlib and py3k. doing
        ##    from matplotlib import collections
        ## gives python's stdlib collections module. explicitly re-importing
        ## the module fixes this.
        # 对于 matplotlib 和 Python 3k 中的导入问题，重新导入子模块 matplotlib.collections 以解决命名冲突
        from_list = import_kwargs.get('fromlist', ())
        for submod in from_list:
            if submod == 'collections' and mod.__name__ == 'matplotlib':
                __import__(module + '.' + submod)
    except ImportError:
        # 捕获 ImportError 异常，如果指定模块未安装，根据 warn_not_installed 参数发出警告
        if warn_not_installed:
            warnings.warn("%s module is not installed" % module, UserWarning,
                    stacklevel=2)
        return
    except catch as e:
        # 捕获 catch 指定的异常类型，如果捕获到异常，则根据 warn_not_installed 参数发出警告
        if warn_not_installed:
            warnings.warn(
                "%s module could not be used (%s)" % (module, repr(e)),
                stacklevel=2)
        return

    if min_module_version:
        # 如果指定了最低模块版本要求，则获取模块的版本信息
        modversion = getattr(mod, module_version_attr)
        if module_version_attr_call_args is not None:
            modversion = modversion(*module_version_attr_call_args)
        # 比较模块的版本与最低要求版本的大小
        if version_tuple(modversion) < version_tuple(min_module_version):
            if warn_old_version:
                # 如果模块版本过旧，根据 min_module_version 的类型生成版本字符串提示
                if isinstance(min_module_version, str):
                    verstr = min_module_version
                elif isinstance(min_module_version, (tuple, list)):
                    verstr = '.'.join(map(str, min_module_version))
                else:
                    verstr = str(min_module_version)
                # 发出版本过旧的警告
                warnings.warn("%s version is too old to use "
                    "(%s or newer required)" % (module, verstr),
                    UserWarning, stacklevel=2)
            return

    # 返回成功导入的模块对象
    return mod
```