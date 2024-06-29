# `.\numpy\numpy\distutils\core.py`

```py
# 导入 sys 模块，用于系统相关操作
import sys
# 从 distutils.core 模块导入 Distribution 类，用于处理发行版相关任务
from distutils.core import Distribution

# 检查是否已导入 setuptools 模块
if 'setuptools' in sys.modules:
    have_setuptools = True
    # 如果有 setuptools，则从 setuptools 中导入 setup 函数，并重命名为 old_setup
    from setuptools import setup as old_setup
    # 导入 setuptools.command 中的 easy_install 模块，用于安装包
    # 注意：easy_install 还会导入 math 模块，可能从当前工作目录中获取
    from setuptools.command import easy_install
    try:
        # 尝试导入 setuptools.command 中的 bdist_egg 模块，用于创建 egg 包
        from setuptools.command import bdist_egg
    except ImportError:
        # 如果导入失败，表示 setuptools 版本过旧，设置 have_setuptools 为 False
        have_setuptools = False
else:
    # 如果没有导入 setuptools，则从 distutils.core 中导入 setup 函数，并重命名为 old_setup
    from distutils.core import setup as old_setup
    have_setuptools = False

# 导入 warnings 模块，用于处理警告信息
import warnings
# 导入 distutils.core 和 distutils.dist 模块，用于核心任务和发行版任务
import distutils.core
import distutils.dist

# 从 numpy.distutils.extension 模块中导入 Extension 类，用于编译扩展模块
from numpy.distutils.extension import Extension  # noqa: F401
# 从 numpy.distutils.numpy_distribution 模块中导入 NumpyDistribution 类，用于处理 NumPy 发行版
from numpy.distutils.numpy_distribution import NumpyDistribution
# 从 numpy.distutils.command 模块中导入多个命令类，用于编译、构建等任务
from numpy.distutils.command import config, config_compiler, \
     build, build_py, build_ext, build_clib, build_src, build_scripts, \
     sdist, install_data, install_headers, install, bdist_rpm, \
     install_clib
# 从 numpy.distutils.misc_util 模块中导入辅助函数 is_sequence 和 is_string
from numpy.distutils.misc_util import is_sequence, is_string

# 定义一个字典 numpy_cmdclass，存储各个命令类对应的处理类
numpy_cmdclass = {'build':            build.build,
                  'build_src':        build_src.build_src,
                  'build_scripts':    build_scripts.build_scripts,
                  'config_cc':        config_compiler.config_cc,
                  'config_fc':        config_compiler.config_fc,
                  'config':           config.config,
                  'build_ext':        build_ext.build_ext,
                  'build_py':         build_py.build_py,
                  'build_clib':       build_clib.build_clib,
                  'sdist':            sdist.sdist,
                  'install_data':     install_data.install_data,
                  'install_headers':  install_headers.install_headers,
                  'install_clib':     install_clib.install_clib,
                  'install':          install.install,
                  'bdist_rpm':        bdist_rpm.bdist_rpm,
                  }

# 如果有 setuptools 模块，则继续进行以下设置
if have_setuptools:
    # 从 numpy.distutils.command 中导入 develop 和 egg_info 模块
    from numpy.distutils.command import develop, egg_info
    # 向 numpy_cmdclass 字典中添加更多命令类和对应的处理类
    numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    numpy_cmdclass['develop'] = develop.develop
    numpy_cmdclass['easy_install'] = easy_install.easy_install
    numpy_cmdclass['egg_info'] = egg_info.egg_info

def _dict_append(d, **kws):
    """向字典 d 中的键值对进行追加或更新
    
    Args:
        d (dict): 目标字典
        **kws: 关键字参数，键为要追加或更新的字典键，值为要追加或更新的对应值
    """
    for k, v in kws.items():
        if k not in d:
            d[k] = v
            continue
        dv = d[k]
        if isinstance(dv, tuple):
            d[k] = dv + tuple(v)
        elif isinstance(dv, list):
            d[k] = dv + list(v)
        elif isinstance(dv, dict):
            _dict_append(dv, **v)
        elif is_string(dv):
            d[k] = dv + v
        else:
            raise TypeError(repr(type(dv)))

def _command_line_ok(_cache=None):
    """检查命令行是否不包含任何帮助或显示请求
    
    Args:
        _cache (list, optional): 用于缓存结果的列表
    
    Returns:
        bool: 如果命令行没有包含帮助或显示请求，则返回 True，否则返回 False
    """
    if _cache:
        return _cache[0]
    elif _cache is None:
        _cache = []
    # 初始化一个布尔变量，用于标记参数是否合法，默认为True
    ok = True
    # 根据 Distribution 类的 display_option_names 属性生成展示选项列表
    display_opts = ['--'+n for n in Distribution.display_option_names]
    # 遍历 Distribution 类的 display_options 属性
    for o in Distribution.display_options:
        # 如果选项的第二个元素为真，将其简写形式添加到展示选项列表中
        if o[1]:
            display_opts.append('-'+o[1])
    # 遍历命令行参数列表
    for arg in sys.argv:
        # 如果命令行参数以 '--help' 开头，或者是 '-h'，或者在展示选项列表中
        if arg.startswith('--help') or arg=='-h' or arg in display_opts:
            # 将 ok 设置为 False，表示参数不合法
            ok = False
            # 跳出循环，不再检查后续的命令行参数
            break
    # 将检查结果添加到 _cache 列表中
    _cache.append(ok)
    # 返回参数合法性的检查结果
    return ok
def get_distribution(always=False):
    # 获取当前的分发对象
    dist = distutils.core._setup_distribution
    # XXX Hack to get numpy installable with easy_install.
    # The problem is easy_install runs it's own setup(), which
    # sets up distutils.core._setup_distribution. However,
    # when our setup() runs, that gets overwritten and lost.
    # We can't use isinstance, as the DistributionWithoutHelpCommands
    # class is local to a function in setuptools.command.easy_install
    # 这段代码用来处理使用 easy_install 安装 numpy 的问题。
    # easy_install 会运行自己的 setup()，设置了 distutils.core._setup_distribution。
    # 当我们的 setup() 运行时，会覆盖并丢失这个设置。通过这段代码来修复这个问题。
    if dist is not None and \
            'DistributionWithoutHelpCommands' in repr(dist):
        dist = None
    if always and dist is None:
        # 如果指定了 always=True 并且 dist 为 None，则使用 NumpyDistribution 类
        dist = NumpyDistribution()
    return dist

def setup(**attr):

    cmdclass = numpy_cmdclass.copy()

    new_attr = attr.copy()
    if 'cmdclass' in new_attr:
        # 更新 cmdclass，如果在 attr 中指定了 cmdclass
        cmdclass.update(new_attr['cmdclass'])
    new_attr['cmdclass'] = cmdclass

    if 'configuration' in new_attr:
        # To avoid calling configuration if there are any errors
        # or help request in command in the line.
        # 如果命令行中存在错误或者帮助请求，避免调用 configuration 方法。
        configuration = new_attr.pop('configuration')

        old_dist = distutils.core._setup_distribution
        old_stop = distutils.core._setup_stop_after
        distutils.core._setup_distribution = None
        distutils.core._setup_stop_after = "commandline"
        try:
            # 递归调用 setup()，处理新的属性
            dist = setup(**new_attr)
        finally:
            # 恢复原来的 _setup_distribution 和 _setup_stop_after 设置
            distutils.core._setup_distribution = old_dist
            distutils.core._setup_stop_after = old_stop
        if dist.help or not _command_line_ok():
            # 如果显示了帮助信息或者命令行状态不正确，直接返回 dist
            # 跳过运行任何命令
            return dist

        # create setup dictionary and append to new_attr
        # 创建配置字典并添加到 new_attr
        config = configuration()
        if hasattr(config, 'todict'):
            config = config.todict()
        _dict_append(new_attr, **config)

    # Move extension source libraries to libraries
    # 将扩展模块的源代码库移动到 libraries 中
    libraries = []
    for ext in new_attr.get('ext_modules', []):
        new_libraries = []
        for item in ext.libraries:
            if is_sequence(item):
                lib_name, build_info = item
                _check_append_ext_library(libraries, lib_name, build_info)
                new_libraries.append(lib_name)
            elif is_string(item):
                new_libraries.append(item)
            else:
                raise TypeError("invalid description of extension module "
                                "library %r" % (item,))
        ext.libraries = new_libraries
    if libraries:
        if 'libraries' not in new_attr:
            new_attr['libraries'] = []
        for item in libraries:
            _check_append_library(new_attr['libraries'], item)

    # sources in ext_modules or libraries may contain header files
    # ext_modules 或 libraries 中可能包含头文件
    if ('ext_modules' in new_attr or 'libraries' in new_attr) \
       and 'headers' not in new_attr:
        new_attr['headers'] = []

    # Use our custom NumpyDistribution class instead of distutils' one
    # 使用我们自定义的 NumpyDistribution 类，替代 distutils 的类
    new_attr['distclass'] = NumpyDistribution

    return old_setup(**new_attr)
# 检查并向库列表中追加新的库项或更新现有库项的构建信息（如果存在）
def _check_append_library(libraries, item):
    # 遍历库列表中的每一项
    for libitem in libraries:
        # 检查当前库项是否为序列（元组或列表）
        if is_sequence(libitem):
            # 如果待添加的项也是序列
            if is_sequence(item):
                # 如果两个序列的第一个元素相等
                if item[0] == libitem[0]:
                    # 如果第二个元素也相等，则表示已存在相同的库项
                    if item[1] is libitem[1]:
                        return  # 直接返回，无需更新
                    # 如果第二个元素不相等，则发出警告
                    warnings.warn("[0] libraries list contains %r with"
                                  " different build_info" % (item[0],),
                                  stacklevel=2)
                    break
            else:
                # 如果待添加的项不是序列，但与库项的第一个元素相等
                if item == libitem[0]:
                    # 发出相应的警告
                    warnings.warn("[1] libraries list contains %r with"
                                  " no build_info" % (item[0],),
                                  stacklevel=2)
                    break
        else:
            # 如果库项本身不是序列，检查待添加的项是否为序列
            if is_sequence(item):
                # 如果待添加的项的第一个元素与当前库项相等
                if item[0] == libitem:
                    # 发出相应的警告
                    warnings.warn("[2] libraries list contains %r with"
                                  " no build_info" % (item[0],),
                                  stacklevel=2)
                    break
            else:
                # 如果两者相等，则表示已存在相同的库项
                if item == libitem:
                    return  # 直接返回，无需更新

    # 如果未发现相同的库项，则将新的库项添加到列表末尾
    libraries.append(item)

# 检查并向外部库列表中追加新的库项或更新现有库项的构建信息（如果存在）
def _check_append_ext_library(libraries, lib_name, build_info):
    # 遍历库列表中的每一项
    for item in libraries:
        # 检查当前库项是否为序列（元组或列表）
        if is_sequence(item):
            # 如果库项是一个序列，并且序列的第一个元素与指定的库名相等
            if item[0] == lib_name:
                # 如果序列的第二个元素也相等，则表示已存在相同的库项
                if item[1] is build_info:
                    return  # 直接返回，无需更新
                # 如果第二个元素不相等，则发出警告
                warnings.warn("[3] libraries list contains %r with"
                              " different build_info" % (lib_name,),
                              stacklevel=2)
                break
        else:
            # 如果当前库项不是序列，并且与指定的库名相等
            if item == lib_name:
                # 发出相应的警告
                warnings.warn("[4] libraries list contains %r with"
                              " no build_info" % (lib_name,),
                              stacklevel=2)
                break

    # 如果未发现相同的库项，则将新的库名和构建信息添加为元组到列表末尾
    libraries.append((lib_name, build_info))
```